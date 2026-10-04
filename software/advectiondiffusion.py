#!/usr/bin/env python3 ##
"""
4 March 2026
Peeyush Khare Ph.D. 
Institute of Climate and Energy Systems (ICE-3)
Forschungszentrum Juelich, Germany

Tracer X GIF Generator (Advection + Turbulent Diffusion)

Two mechanical effects taken into consideration:
- Advection: particles move with a mean wind field u(x,y), v(x,y)
- Turbulent diffusion: random-walk with eddy diffusivity K:
    dx = u*dt + sqrt(2*K*dt)*N(0,1)
    dy = v*dt + sqrt(2*K*dt)*N(0,1)

This script provides a small Tkinter GUI to generate an animated GIF.

Dependencies:
  pip install matplotlib pillow

Run protocol:
  python advection_diffusion_gif_gui.py
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
matplotlib.use("Agg")  # render offscreen (works headless too)
import matplotlib.pyplot as plt


# ----------------------------
# Simulation protocols
# ----------------------------

@dataclass
class SimParams:
    # domain (data coords) - defaults for canvas but the user overrides in the GUI
    xmin: float = -1.0
    xmax: float =  1.0
    ymin: float =  0.0
    ymax: float =  1.0

    # time
    duration_s: float = 12.0  #duration of simulation
    fps: int = 20             #sets the number of frames per second
    dt: float = 0.05  # internal integrator time step (s)

    # source
    source_x: float = 0.0
    source_y: float = 0.08
    emission_rate: float = 60.0   # particles / second
    lifetime_s: float = 8.0       # mean lifetime (s)
    max_particles: int = 4000

    # physics
    wind_u: float = 0.18      # mean u (x) in domain units/s
    wind_v: float = 0.05      # mean v (y) in domain units/s
    K: float = 0.015          # eddy diffusivity in domain units^2/s

    # rendering
    marker_size: float = 8.0
    base_alpha: float = 0.55
    bgcolor: str = "black"
    fgcolor: str = "white"
    title: str = "Tracer X: advection + turbulent diffusion"


class Particle:
    __slots__ = ("x", "y", "age", "life")
    def __init__(self, x: float, y: float, life: float):
        self.x = x
        self.y = y
        self.age = 0.0
        self.life = life

# This function sets the wind field as wind shear varies with height above the surface
def wind_field(x: float, y: float, p: SimParams) -> tuple[float, float]:
    """
    Example wind field: base wind + mild vertical shear + gentle vertical meander.
    Replace this with any u(x,y), v(x,y) you like.
    """
    # shear increases with height
    shear = 0.7 + 0.6 * (y / max(1e-9, p.ymax))
    u = p.wind_u * shear

    # small oscillation makes the GIF more visually informative
    v = p.wind_v + 0.02 * math.sin(2.0 * math.pi * (x + 0.2)) * (0.3 + 0.7 * y)
    return u, v


def simulate_and_render_gif(out_path: str, p: SimParams) -> None:
    frames = int(max(1, round(p.duration_s * p.fps)))
    frame_dt = 1.0 / p.fps

    # Integrate with smaller steps than frame_dt for stability/smoothness
    steps_per_frame = max(1, int(round(frame_dt / p.dt)))
    dt = frame_dt / steps_per_frame

    particles: list[Particle] = []
    images: list[Image.Image] = []

    # Figure setup (consistent size)
    dpi = 120
    fig_w, fig_h = 6.0, 3.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    def style_axes():
        ax.set_xlim(p.xmin, p.xmax)
        ax.set_ylim(p.ymin, p.ymax)
        ax.set_facecolor(p.bgcolor)
        fig.patch.set_facecolor(p.bgcolor)
        ax.tick_params(colors=p.fgcolor, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(p.fgcolor)

    style_axes()

    spawn_accum = 0.0  # allows fractional particles per time step

    for fi in range(frames):
        # integrate between frames
        for _ in range(steps_per_frame):
            # spawn new particles. This is user defined in the GUI
            spawn_accum += p.emission_rate * dt
            n_new = int(spawn_accum)
            if n_new > 0:
                spawn_accum -= n_new
                for _ in range(n_new):
                    if len(particles) >= p.max_particles:
                        break
                    sx = p.source_x + random.gauss(0.0, 0.01)
                    sy = p.source_y + random.gauss(0.0, 0.01)
                    life = max(0.5, random.gauss(p.lifetime_s, 0.8))
                    particles.append(Particle(sx, sy, life))

            # step particles
            K = max(0.0, p.K)
            sigma = math.sqrt(2.0 * K * dt) if K > 0 else 0.0

            alive: list[Particle] = []
            for prt in particles:
                u, v = wind_field(prt.x, prt.y, p)
                prt.x += u * dt + sigma * random.gauss(0.0, 1.0)
                prt.y += v * dt + sigma * random.gauss(0.0, 1.0)
                prt.age += dt

                if (prt.age < prt.life and
                    (p.xmin <= prt.x <= p.xmax) and
                    (p.ymin <= prt.y <= p.ymax)):
                    alive.append(prt)
            particles = alive

        # render
        ax.clear()
        style_axes()
        ax.set_title(p.title, color=p.fgcolor, fontsize=11, pad=8)

        # Draw chimney/stack LAST so it cannot be hidden by particles/text.
        # Use min/max so it still draws even if source_y < ymin in the GUI.
        y0 = min(p.ymin, p.source_y)
        y1 = max(p.ymin, p.source_y)

        # Only draw if it has visible height
        if (y1 - y0) > 1e-6:
            # Thickness scales with domain size so it stays visible when you change limits
            lw = max(2.0, 0.01 * (p.xmax - p.xmin) * 120.0 / 6.0)  # rough scale for your figure
            ax.plot(
                [p.source_x, p.source_x],
                [y0, y1],
                color="white",
                linewidth=lw,
                alpha=1.0,
                zorder=100,
                solid_capstyle="butt",
            )

        if particles:
            xs = [q.x for q in particles]
            ys = [q.y for q in particles]

            # Fade with age: younger particles brighter
            rgba = []
            for q in particles:
                a = max(0.05, p.base_alpha * (1.0 - q.age / max(1e-9, q.life)))
                rgba.append((1.0, 1.0, 1.0, a))

            ax.scatter(xs, ys, s=p.marker_size, c=rgba, marker="o", linewidths=0)

            # Label a subset as "X" (keeps it readable)
            newest = sorted(particles, key=lambda z: z.age)[:120]
            for q in newest:
                a = max(0.08, p.base_alpha * (1.0 - q.age / max(1e-9, q.life)))
                ax.text(q.x, q.y, "X", color=(1, 1, 1, a), fontsize=8,
                        ha="center", va="center")

        t = fi * frame_dt
        ax.text(
            0.01, 0.02,
            f"t={t:4.1f}s   wind=({p.wind_u:.3f},{p.wind_v:.3f})   K={p.K:.4f}   N={len(particles)}",
            transform=ax.transAxes, color=p.fgcolor, fontsize=8,
            ha="left", va="bottom"
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


# ----------------------------
# GUI dispatch
# ----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tracer X GIF – advection + turbulent diffusion")
        self.geometry("640x680")
        self.resizable(True, True)

        self.params = SimParams()

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Simulation parameters", font=("TkDefaultFont", 12, "bold"))\
            .grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        self.vars: dict[str, tk.Variable] = {}

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
        add_spin(r, "Emission rate (particles/s)", "emission_rate", 1, 500, 5); r += 1
        add_spin(r, "Lifetime mean (s)", "lifetime_s", 0.5, 30, 0.5); r += 1
        add_spin(r, "Max particles", "max_particles", 200, 20000, 200, is_int=True); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        add_spin(r, "Wind u (x) (units/s)", "wind_u", -1.0, 1.0, 0.01); r += 1
        add_spin(r, "Wind v (y) (units/s)", "wind_v", -1.0, 1.0, 0.01); r += 1
        add_spin(r, "Eddy diffusivity K (units²/s)", "K", 0.0, 0.2, 0.001); r += 1

        ttk.Separator(root).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10); r += 1

        # Domain limits
        add_spin(r, "Domain xmin", "xmin", -100.0, 100.0, 0.5); r += 1
        add_spin(r, "Domain xmax", "xmax", -100.0, 100.0, 0.5); r += 1
        add_spin(r, "Domain ymin", "ymin", -100.0, 100.0, 0.5); r += 1
        add_spin(r, "Domain ymax", "ymax", -100.0, 100.0, 0.5); r += 1


        add_spin(r, "Source x", "source_x", -1.0, 1.0, 0.01); r += 1
        add_spin(r, "Source y", "source_y", 0.0, 1.0, 0.01); r += 1
        add_spin(r, "Marker size", "marker_size", 1, 30, 1); r += 1
        add_spin(r, "Base alpha", "base_alpha", 0.05, 1.0, 0.05); r += 1

        ttk.Label(root, text="Output").grid(row=r, column=0, sticky="w", pady=(14, 4))
        self.out_entry = ttk.Entry(root, width=50)
        self.out_entry.grid(row=r, column=1, columnspan=2, sticky="w", pady=(14, 4))
        self.out_entry.insert(0, os.path.join(os.getcwd(), "tracer_X.gif"))
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
            initialfile="tracer_X.gif",
        )
        if path:
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, path)

    def read_params(self) -> SimParams:
        p = SimParams()
        for k, var in self.vars.items():
            if isinstance(var, tk.IntVar):
                setattr(p, k, int(var.get()))
            else:
                setattr(p, k, float(var.get()))
        return p

    def generate(self):
        out_path = self.out_entry.get().strip()
        if not out_path:
            messagebox.showerror("No output path", "Please choose an output path for the GIF.")
            return
        try:
            p = self.read_params()
            self.status.set("Generating GIF… (long durations / high FPS can take a bit)")
            self.update_idletasks()
            simulate_and_render_gif(out_path, p)
            self.status.set(f"Done: {out_path}")
            messagebox.showinfo("Done", f"Saved GIF:\\n{out_path}")
        except Exception as e:
            self.status.set("Error.")
            messagebox.showerror("Error while generating", str(e))


def main():
    # Optional: better DPI handling on Windows
    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
