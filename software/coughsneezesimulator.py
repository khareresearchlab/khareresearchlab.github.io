#!/usr/bin/env python3
"""
Room Aerosol GIF Generator (cough / sneeze / breathing)

Modified from the user's advection_diffusion_gui_main.py.
This version simulates a short indoor aerosol puff with:
- pulse release (cough / sneeze / breathing presets)
- initial jet momentum from the mouth
- indoor ventilation / recirculation flow
- turbulent diffusion via random walk
- optional gravitational settling
- wall reflection with damping
- first-order removal by ventilation / deposition

Dependencies:
  pip install matplotlib pillow

Run:
  python room_aerosol_cough_sneeze_simulator.py
"""
from __future__ import annotations

import os
import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, filedialog, messagebox

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SimParams:
    # room domain (meters) - 2D cross section
    xmin: float = 0.0
    xmax: float = 5.0
    ymin: float = 0.0
    ymax: float = 3.0

    # time
    duration_s: float = 12.0
    fps: int = 20
    dt: float = 0.02

    # source / release
    source_x: float = 0.8
    source_y: float = 1.55
    release_start_s: float = 0.5
    release_duration_s: float = 0.25
    total_particles: int = 1500
    lifetime_s: float = 12.0
    max_particles: int = 15000

    # cough / sneeze jet
    jet_speed: float = 3.0            # m/s
    jet_angle_deg: float = 0.0        # 0 = horizontal to the right
    jet_spread_deg: float = 10.0      # angular spread
    jet_relaxation_s: float = 0.18    # momentum decay toward room air
    source_spread_m: float = 0.025    # mouth/source spread

    # room airflow and mixing
    flow_mode: str = "uniform"        # uniform or recirculation
    wind_u: float = 0.08              # m/s
    wind_v: float = 0.0               # m/s
    recirc_strength: float = 0.08     # 1/s-ish visual parameter
    K: float = 0.01                   # eddy diffusivity m^2/s

    # aerosol physics
    settling_v: float = 0.003         # m/s downward
    removal_rate_s: float = 0.0008    # 1/s, ventilation + deposition
    bounce_damping_x: float = 0.30
    bounce_damping_y: float = 0.20

    # rendering
    marker_size: float = 10.0
    base_alpha: float = 0.55
    bgcolor: str = "black"
    fgcolor: str = "white"
    title: str = "Indoor aerosol transport after cough / sneeze"


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "age", "life")

    def __init__(self, x: float, y: float, vx: float, vy: float, life: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.age = 0.0
        self.life = life


def wind_field(x: float, y: float, p: SimParams) -> tuple[float, float]:
    """Indoor ventilation field."""
    if p.flow_mode == "recirculation":
        xc = 0.5 * (p.xmin + p.xmax)
        yc = 0.5 * (p.ymin + p.ymax)
        u = p.wind_u + p.recirc_strength * (y - yc)
        v = p.wind_v - p.recirc_strength * (x - xc)
        return u, v
    return p.wind_u, p.wind_v


def apply_wall_interactions(prt: Particle, p: SimParams) -> bool:
    """Reflect particles from room walls. Return False if particle should be removed."""
    # left / right walls
    if prt.x < p.xmin:
        prt.x = p.xmin + (p.xmin - prt.x)
        prt.vx = abs(prt.vx) * p.bounce_damping_x
    elif prt.x > p.xmax:
        prt.x = p.xmax - (prt.x - p.xmax)
        prt.vx = -abs(prt.vx) * p.bounce_damping_x

    # floor / ceiling
    if prt.y < p.ymin:
        # treat floor as deposition-dominant: most particles are removed
        if random.random() < 0.85:
            return False
        prt.y = p.ymin + (p.ymin - prt.y)
        prt.vy = abs(prt.vy) * p.bounce_damping_y
    elif prt.y > p.ymax:
        prt.y = p.ymax - (prt.y - p.ymax)
        prt.vy = -abs(prt.vy) * p.bounce_damping_y

    return True


def spawn_particles(particles: list[Particle], p: SimParams, t_now: float, dt: float, spawn_accum: float) -> float:
    """Spawn a pulse release between release_start_s and release_start_s + release_duration_s."""
    if p.release_duration_s <= 0:
        return spawn_accum

    release_end = p.release_start_s + p.release_duration_s
    if p.release_start_s <= t_now < release_end:
        emit_rate = p.total_particles / p.release_duration_s
        spawn_accum += emit_rate * dt
        n_new = int(spawn_accum)
        if n_new > 0:
            spawn_accum -= n_new
            for _ in range(n_new):
                if len(particles) >= p.max_particles:
                    break
                sx = p.source_x + random.gauss(0.0, p.source_spread_m)
                sy = p.source_y + random.gauss(0.0, 0.5 * p.source_spread_m)

                theta = math.radians(p.jet_angle_deg + random.gauss(0.0, p.jet_spread_deg))
                speed = max(0.05, random.gauss(p.jet_speed, 0.18 * max(0.2, p.jet_speed)))
                vx0 = speed * math.cos(theta)
                vy0 = speed * math.sin(theta)
                life = max(0.8, random.gauss(p.lifetime_s, 0.15 * p.lifetime_s))
                particles.append(Particle(sx, sy, vx0, vy0, life))
    return spawn_accum


def simulate_and_render_gif(out_path: str, p: SimParams) -> None:
    if p.xmax <= p.xmin or p.ymax <= p.ymin:
        raise ValueError("Domain limits are invalid. Make sure xmax > xmin and ymax > ymin.")
    if p.fps <= 0 or p.dt <= 0 or p.duration_s <= 0:
        raise ValueError("duration_s, fps, and dt must be positive.")

    frames = int(max(1, round(p.duration_s * p.fps)))
    frame_dt = 1.0 / p.fps
    steps_per_frame = max(1, int(round(frame_dt / p.dt)))
    dt = frame_dt / steps_per_frame

    particles: list[Particle] = []
    images: list[Image.Image] = []
    spawn_accum = 0.0
    t_now = 0.0

    dpi = 120
    fig_w, fig_h = 7.2, 4.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    def style_axes() -> None:
        ax.set_xlim(p.xmin, p.xmax)
        ax.set_ylim(p.ymin, p.ymax)
        ax.set_facecolor(p.bgcolor)
        fig.patch.set_facecolor(p.bgcolor)
        ax.tick_params(colors=p.fgcolor, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(p.fgcolor)
        ax.set_xlabel("x (m)", color=p.fgcolor)
        ax.set_ylabel("y (m)", color=p.fgcolor)

    style_axes()

    for fi in range(frames):
        for _ in range(steps_per_frame):
            spawn_accum = spawn_particles(particles, p, t_now, dt, spawn_accum)

            K = max(0.0, p.K)
            sigma = math.sqrt(2.0 * K * dt) if K > 0 else 0.0
            alive: list[Particle] = []

            for prt in particles:
                u_room, v_room = wind_field(prt.x, prt.y, p)

                tau = max(1e-6, p.jet_relaxation_s)
                prt.vx += (u_room - prt.vx) * dt / tau
                prt.vy += (v_room - prt.vy) * dt / tau

                prt.x += prt.vx * dt + sigma * random.gauss(0.0, 1.0)
                prt.y += (prt.vy - p.settling_v) * dt + sigma * random.gauss(0.0, 1.0)
                prt.age += dt

                # first-order removal by ventilation / deposition / filtration
                if p.removal_rate_s > 0 and random.random() < p.removal_rate_s * dt:
                    continue

                if prt.age >= prt.life:
                    continue

                if not apply_wall_interactions(prt, p):
                    continue

                alive.append(prt)

            particles = alive
            t_now += dt

        ax.clear()
        style_axes()
        ax.set_title(p.title, color=p.fgcolor, fontsize=11, pad=8)

        # room outline emphasis
        ax.plot([p.xmin, p.xmax, p.xmax, p.xmin, p.xmin],
                [p.ymin, p.ymin, p.ymax, p.ymax, p.ymin],
                color=(1, 1, 1, 0.4), linewidth=1.2)

        # source marker (mouth)
        ax.plot(p.source_x, p.source_y, marker="o", markersize=6, color="white")
        ax.text(p.source_x + 0.08, p.source_y + 0.05, "mouth", color=p.fgcolor, fontsize=8)

        # optional indication of mean ventilation direction
        mid_y = 0.88 * p.ymax + 0.12 * p.ymin
        ax.arrow(p.xmin + 0.15, mid_y, 0.45, 0.0, color="white", width=0.006,
                 head_width=0.08, head_length=0.12, length_includes_head=True, alpha=0.55)
        ax.text(p.xmin + 0.65, mid_y + 0.05, "airflow", color=p.fgcolor, fontsize=8)

        if particles:
            xs = [q.x for q in particles]
            ys = [q.y for q in particles]
            rgba = []
            for q in particles:
                a = max(0.04, p.base_alpha * (1.0 - q.age / max(1e-9, q.life)))
                rgba.append((1.0, 1.0, 1.0, a))
            ax.scatter(xs, ys, s=p.marker_size, c=rgba, marker="o", linewidths=0)

            newest = sorted(particles, key=lambda z: z.age)[:80]
            for q in newest:
                a = max(0.08, p.base_alpha * (1.0 - q.age / max(1e-9, q.life)))
                ax.text(q.x, q.y, "•", color=(1, 1, 1, a), fontsize=8, ha="center", va="center")

        ax.text(
            0.01, 0.02,
            (
                f"t={fi * frame_dt:4.1f}s   N={len(particles)}   "
                f"U=({p.wind_u:.2f},{p.wind_v:.2f}) m/s   K={p.K:.3f} m²/s   "
                f"settling={p.settling_v:.3f} m/s"
            ),
            transform=ax.transAxes,
            color=p.fgcolor,
            fontsize=8,
            ha="left",
            va="bottom",
        )

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im = Image.frombuffer(
            "RGBA",
            fig.canvas.get_width_height(),
            buf,
            "raw",
            "RGBA",
            0,
            1,
        ).convert("RGB")
        images.append(im)

    plt.close(fig)

    if not images:
        raise RuntimeError("No frames were rendered.")

    duration_ms = int(round(1000 / p.fps))
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Indoor Aerosol GIF – cough / sneeze / breathing")
        self.geometry("760x900")
        self.resizable(True, True)

        self.params = SimParams()

        # --- Scrollable container ---
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        scrollable_frame = ttk.Frame(canvas, padding=12)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        root = scrollable_frame

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(root, text="Indoor aerosol simulation parameters", font=("TkDefaultFont", 12, "bold")) \
            .grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        self.vars: dict[str, tk.Variable] = {}
        self.flow_var = tk.StringVar(value=self.params.flow_mode)

        def add_spin(row: int, label: str, attr: str, from_: float, to: float, step: float, is_int: bool = False):
            ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", pady=4)
            if is_int:
                var = tk.IntVar(value=int(getattr(self.params, attr)))
            else:
                var = tk.DoubleVar(value=float(getattr(self.params, attr)))
            self.vars[attr] = var
            sp = ttk.Spinbox(root, textvariable=var, from_=from_, to=to, increment=step, width=14)
            sp.grid(row=row, column=1, sticky="w", pady=4)
            ttk.Label(root, text=f"({attr})").grid(row=row, column=2, sticky="w", pady=4)

        r = 1
        add_spin(r, "Duration (s)", "duration_s", 1, 60, 1); r += 1
        add_spin(r, "FPS", "fps", 5, 60, 1, is_int=True); r += 1
        add_spin(r, "Integrator dt (s)", "dt", 0.005, 0.2, 0.005); r += 1
        add_spin(r, "Lifetime mean (s)", "lifetime_s", 1, 60, 0.5); r += 1
        add_spin(r, "Max particles", "max_particles", 200, 50000, 200, is_int=True); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Release start (s)", "release_start_s", 0, 20, 0.1); r += 1
        add_spin(r, "Release duration (s)", "release_duration_s", 0.01, 10, 0.05); r += 1
        add_spin(r, "Total particles", "total_particles", 10, 50000, 50, is_int=True); r += 1
        add_spin(r, "Source x (m)", "source_x", 0.0, 20.0, 0.01); r += 1
        add_spin(r, "Source y (m)", "source_y", 0.0, 10.0, 0.01); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Jet speed (m/s)", "jet_speed", 0.0, 50.0, 0.1); r += 1
        add_spin(r, "Jet angle (deg)", "jet_angle_deg", -90.0, 90.0, 1.0); r += 1
        add_spin(r, "Jet spread (deg)", "jet_spread_deg", 0.0, 45.0, 1.0); r += 1
        add_spin(r, "Jet relaxation (s)", "jet_relaxation_s", 0.01, 2.0, 0.01); r += 1
        add_spin(r, "Source spread (m)", "source_spread_m", 0.001, 0.2, 0.001); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Mean airflow Ux (m/s)", "wind_u", -2.0, 2.0, 0.01); r += 1
        add_spin(r, "Mean airflow Uy (m/s)", "wind_v", -2.0, 2.0, 0.01); r += 1
        add_spin(r, "Recirculation strength", "recirc_strength", 0.0, 1.0, 0.01); r += 1
        add_spin(r, "Eddy diffusivity K (m²/s)", "K", 0.0, 0.5, 0.001); r += 1
        add_spin(r, "Settling velocity (m/s)", "settling_v", 0.0, 0.1, 0.0005); r += 1
        add_spin(r, "Removal rate (1/s)", "removal_rate_s", 0.0, 0.1, 0.0002); r += 1

        ttk.Label(root, text="Flow mode").grid(row=r, column=0, sticky="w", pady=4)
        flow_box = ttk.Combobox(root, textvariable=self.flow_var, width=18, state="readonly",
                                values=["uniform", "recirculation"])
        flow_box.grid(row=r, column=1, sticky="w", pady=4)
        ttk.Label(root, text="(flow_mode)").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Room xmin (m)", "xmin", -5.0, 20.0, 0.1); r += 1
        add_spin(r, "Room xmax (m)", "xmax", 0.1, 30.0, 0.1); r += 1
        add_spin(r, "Room ymin (m)", "ymin", -5.0, 10.0, 0.1); r += 1
        add_spin(r, "Room ymax (m)", "ymax", 0.1, 15.0, 0.1); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Marker size", "marker_size", 1, 30, 1); r += 1
        add_spin(r, "Base alpha", "base_alpha", 0.05, 1.0, 0.05); r += 1

        ttk.Label(root, text="Presets").grid(row=r, column=0, sticky="w", pady=(10, 4))
        presets = ttk.Frame(root)
        presets.grid(row=r, column=1, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Button(presets, text="Cough", command=lambda: self.apply_preset("cough")).pack(side="left", padx=(0, 6))
        ttk.Button(presets, text="Sneeze", command=lambda: self.apply_preset("sneeze")).pack(side="left", padx=6)
        ttk.Button(presets, text="Breathing", command=lambda: self.apply_preset("breathing")).pack(side="left", padx=6)
        ttk.Button(presets, text="Reset indoor default", command=lambda: self.apply_preset("default")).pack(side="left", padx=6)
        r += 1

        ttk.Label(root, text="Output").grid(row=r, column=0, sticky="w", pady=(14, 4))
        self.out_entry = ttk.Entry(root, width=50)
        self.out_entry.grid(row=r, column=1, columnspan=2, sticky="w", pady=(14, 4))
        self.out_entry.insert(0, os.path.join(os.getcwd(), "room_aerosol.gif"))
        ttk.Button(root, text="Browse…", command=self.browse).grid(row=r, column=3, sticky="w", pady=(14, 4))
        r += 1

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status).grid(row=r, column=0, columnspan=4, sticky="w", pady=(10, 0))
        r += 1

        btns = ttk.Frame(root)
        btns.grid(row=r, column=0, columnspan=4, sticky="ew", pady=12)
        ttk.Button(btns, text="Generate GIF", command=self.generate).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        for c in range(4):
            root.grid_columnconfigure(c, weight=1 if c == 1 else 0)

    def browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif")],
            initialfile="room_aerosol.gif",
        )
        if path:
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, path)

    def set_var(self, attr: str, value):
        if attr not in self.vars:
            return
        var = self.vars[attr]
        if isinstance(var, tk.IntVar):
            var.set(int(value))
        else:
            var.set(float(value))

    def apply_preset(self, name: str):
        presets = {
            "default": dict(
                duration_s=12, fps=20, dt=0.02,
                release_start_s=0.5, release_duration_s=0.25, total_particles=1500,
                jet_speed=3.0, jet_angle_deg=0.0, jet_spread_deg=10.0, jet_relaxation_s=0.18,
                wind_u=0.08, wind_v=0.0, K=0.01, settling_v=0.003, removal_rate_s=0.0008,
                source_x=0.8, source_y=1.55, xmin=0.0, xmax=5.0, ymin=0.0, ymax=3.0,
                lifetime_s=12.0
            ),
            "cough": dict(
                release_duration_s=0.25, total_particles=1200,
                jet_speed=4.0, jet_spread_deg=12.0, settling_v=0.003,
                K=0.01, wind_u=0.08, lifetime_s=12.0
            ),
            "sneeze": dict(
                release_duration_s=0.18, total_particles=3500,
                jet_speed=10.0, jet_spread_deg=18.0, settling_v=0.006,
                K=0.012, wind_u=0.08, lifetime_s=10.0
            ),
            "breathing": dict(
                release_duration_s=5.0, total_particles=1400,
                jet_speed=0.8, jet_spread_deg=20.0, settling_v=0.001,
                K=0.008, wind_u=0.05, lifetime_s=14.0
            ),
        }
        vals = presets[name]
        for k, v in vals.items():
            if k == "flow_mode":
                self.flow_var.set(v)
            else:
                self.set_var(k, v)
        self.status.set(f"Preset applied: {name}")

    def read_params(self) -> SimParams:
        p = SimParams()
        for k, var in self.vars.items():
            if isinstance(var, tk.IntVar):
                setattr(p, k, int(var.get()))
            else:
                setattr(p, k, float(var.get()))
        p.flow_mode = self.flow_var.get().strip() or "uniform"
        return p

    def generate(self):
        out_path = self.out_entry.get().strip()
        if not out_path:
            messagebox.showerror("No output path", "Please choose an output path for the GIF.")
            return
        try:
            p = self.read_params()
            self.status.set("Generating GIF…")
            self.update_idletasks()
            simulate_and_render_gif(out_path, p)
            self.status.set(f"Done: {out_path}")
            messagebox.showinfo("Done", f"Saved GIF:\n{out_path}")
        except Exception as e:
            self.status.set("Error.")
            messagebox.showerror("Error while generating", str(e))


def main():
    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
