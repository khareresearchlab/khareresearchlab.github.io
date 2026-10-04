
"""
Fire + smoke + formula emissions GIF (GUI version)

Based on fire_works8_gif5.py, with a PyQt GUI to:
- Paste/edit compound list in a table
- Choose emission geometry: conical or horizontal surface
- Control cone / horizontal dimensions
- Control turbulence and movement speed
- Choose color gradient preset for emissions (text particles)
- Show progress bar while building GIF

Run:
  python fire_works8_gif5_gui.py
"""

from __future__ import annotations

import io
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image


# -----------------------------
# Defaults (kept close to original)
# -----------------------------
W, H = 640, 360
FPS = 30
DURATION_S = 15
FRAMES = FPS * DURATION_S

DEFAULT_GIF_NAME = "fire_formulas.gif"

# Fire region in normalized coords (0..1)
FIRE_X0, FIRE_X1 = 0.32, 0.68
FIRE_Y0, FIRE_Y1 = 0.05, 0.38

# Scene bounds in data coordinates
YMAX_DEFAULT = 2.6
XMIN, XMAX = -0.5, 1.5
SCALE_X = (XMAX - XMIN) / 1.0

# Ember (burnt particle) system (kept)
ENABLE_EMBERS = True
EMBER_RATE = 4.0
MAX_EMBERS = 500
EMBER_SIZE_RANGE = (6, 26)
EMBER_LIFE_RANGE = (1.2, 2.6)
EMBER_VY_RANGE = (0.22, 0.42)
EMBER_VX_RANGE = (-0.10, 0.10)
EMBER_GRAVITY = -0.04
EMBER_FLICKER = 7.0

ENABLE_SMOKE = False  # same as original


# -----------------------------
# Helpers (kept)
# -----------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def make_grid(w, h):
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

X_GRID, Y_GRID = make_grid(W, H)

fire_mask_x = smoothstep(FIRE_X0, FIRE_X0 + 0.06, X_GRID) * (1 - smoothstep(FIRE_X1 - 0.06, FIRE_X1, X_GRID))
fire_mask_y = smoothstep(FIRE_Y0, FIRE_Y0 + 0.08, Y_GRID) * (1 - smoothstep(FIRE_Y1 - 0.10, FIRE_Y1, Y_GRID))
FIRE_MASK = fire_mask_x * fire_mask_y

SMOKE_MASK = (smoothstep(0.10, 0.35, Y_GRID) * (1 - smoothstep(0.55, 0.95, Y_GRID))) * 0.95


def fire_field(t: float):
    base = np.clip((Y_GRID - FIRE_Y0) / (FIRE_Y1 - FIRE_Y0), 0, 1)
    base = (1 - base) ** 0.35

    lateral = 1 - ((X_GRID - 0.5) / 0.22) ** 2
    lateral = np.clip(lateral, 0, 1)

    n1 = np.sin((X_GRID * 9.0 + t * 2.6) * 2 * np.pi) * np.cos((Y_GRID * 6.0 - t * 3.1) * 2 * np.pi)
    n2 = np.sin((X_GRID * 17.0 - t * 4.4) * 2 * np.pi + Y_GRID * 3.0) * 0.6
    n3 = np.cos((X_GRID * 5.0 + Y_GRID * 11.0 + t * 1.9) * 2 * np.pi) * 0.35
    noise = (n1 + n2 + n3) / 2.0

    rise = np.sin((X_GRID * 12.0 + noise * 2.0 + t * 5.0) * 2 * np.pi)
    rise = (rise * 0.5 + 0.5) ** 2.2

    intensity = (0.9 * base * lateral + 0.7 * rise * base) * (0.65 + 0.55 * (noise * 0.5 + 0.5))
    intensity *= FIRE_MASK

    flicker = 0.10 * (math.sin(t * 18.0) + math.sin(t * 7.0 + 1.3)) * 0.5
    intensity = np.clip(intensity * (1.0 + flicker), 0, 1)
    return intensity.astype(np.float32)

def smoke_field(t: float, fire_intensity):
    if not ENABLE_SMOKE:
        return np.zeros((H, W, 4), dtype=np.float32)

    drift = 0.02 * np.sin((X_GRID * 4.0 + t * 0.6) * 2 * np.pi) + 0.02 * np.cos((Y_GRID * 3.0 - t * 0.5) * 2 * np.pi)
    band = np.clip(fire_intensity * 0.8 + (0.22 + drift) * SMOKE_MASK, 0, 1)
    alpha = np.clip(band * (0.34 * (1 - smoothstep(0.45, 0.98, Y_GRID))), 0, 0.42)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = 0.18 * alpha
    rgba[..., 1] = 0.18 * alpha
    rgba[..., 2] = 0.19 * alpha
    rgba[..., 3] = alpha
    return rgba


# -----------------------------
# Color gradients for text
# -----------------------------
GRADIENT_PRESETS = [
    "Flame (blue→red→black)",
    "Viridis",
    "Plasma",
    "Inferno",
    "Magma",
    "Cividis",
    "Greys",
    "Black",
]

def text_color_for_height(y: float, y0: float, ymax: float, preset: str) -> Tuple[float, float, float]:
    # normalize 0..1 by traveled height
    tcol = (y - y0) / max(0.65, (ymax - y0))
    tcol = max(0.0, min(1.0, float(tcol)))

    if preset == "Black":
        return (0.0, 0.0, 0.0)

    if preset == "Flame (blue→red→black)":
        if tcol < 0.35:
            u = tcol / 0.35
            r = u
            g = 0.15 * (1.0 - u)
            b = 1.0 - u
        else:
            u = (tcol - 0.35) / 0.65
            r = 1.0 - u
            g = 0.0
            b = 0.0
        return (float(r), float(g), float(b))

    # matplotlib colormap
    cmap = mpl.colormaps.get(preset.lower(), mpl.colormaps["viridis"])
    r, g, b, _ = cmap(tcol)
    return (float(r), float(g), float(b))


# -----------------------------
# Particle systems
# -----------------------------
@dataclass
class EmissionParams:
    mode: str  # "cone" or "horizontal"
    # shared
    emit_rate: float = 0.3
    max_particles: int = 240
    speed_scale: float = 0.80
    turbulence: float = 1.0
    gradient: str = GRADIENT_PRESETS[0]
    ymax: float = YMAX_DEFAULT

    # cone-specific
    cone_base_w: float = 0.030  # in normalized 0..1, scaled later by SCALE_X
    cone_k: float = 1.20        # growth per unit height, scaled later by SCALE_X
    axis_wander: float = 0.03   # amplitude of plume axis X wander

    # horizontal-specific
    h_xmin: float = 0.40
    h_xmax: float = 0.60
    h_wander: float = 0.10      # stronger lateral wander for horizontal mode


PLUME_AXIS_X = 0.50

class FormulaParticleBase:
    __slots__ = ("text","x","y","vx","vy","life","age","size","y0","seed","phase")

    def __init__(self, text: str, x0: float, y0: float, vx: float, vy: float, life: float, size: float):
        self.text = text
        self.x = x0
        self.y0 = y0
        self.y = y0
        self.vx = vx
        self.vy = vy
        self.life = life
        self.age = 0.0
        self.size = size
        self.seed = random.uniform(0, 1000)
        self.phase = random.uniform(0, 2 * math.pi)

    def step(self, dt: float, params: EmissionParams) -> bool:
        raise NotImplementedError

    @property
    def alpha(self) -> float:
        t = self.age / self.life
        fade_in = min(1.0, t / 0.12)
        fade_out = max(0.0, (1.0 - t) / 0.35)
        return float(min(fade_in, fade_out))


class FormulaParticleCone(FormulaParticleBase):
    __slots__ = ("u","vu","base_w","cone_k")

    def __init__(self, text: str, x0: float, y0: float, vy: float, life: float, size: float, params: EmissionParams):
        super().__init__(text=text, x0=x0, y0=y0, vx=0.0, vy=vy, life=life, size=size)
        # Cone geometry (scaled to scene)
        self.base_w = max(0.0, params.cone_base_w) * SCALE_X
        self.cone_k = max(0.0, params.cone_k) * SCALE_X
        self.u = random.uniform(-0.20, 0.20)
        self.vu = random.uniform(-0.25, 0.25)

    def step(self, dt: float, params: EmissionParams) -> bool:
        global PLUME_AXIS_X
        self.age += dt
        t = self.age

        # rise
        self.y += self.vy * dt * params.speed_scale
        self.vy *= 0.996

        dy = max(0.0, self.y - self.y0)
        half_width = self.base_w + self.cone_k * dy

        # Smooth turbulent forcing in normalized space
        f = (
            0.9 * math.sin(self.seed + 1.4 * t + self.phase) +
            0.6 * math.sin(self.seed * 0.7 + 2.3 * t)
        )
        noise = random.uniform(-0.8, 0.8)
        height_gain = 0.7 + 0.8 * min(1.0, self.y)

        turb = max(0.0, params.turbulence)
        self.vu += turb * (0.55 * f + 0.35 * noise) * dt * height_gain * params.speed_scale
        self.vu *= 0.985
        self.u += self.vu * dt

        # reflect boundaries
        if self.u > 1.0:
            self.u = 1.0 - (self.u - 1.0)
            self.vu *= -0.6
        elif self.u < -1.0:
            self.u = -1.0 - (self.u + 1.0)
            self.vu *= -0.6

        self.x = PLUME_AXIS_X + self.u * half_width

        # fade near top
        top_fade = 1.0 - smoothstep(params.ymax - 0.35, params.ymax - 0.02, self.y)
        return (self.age < self.life) and (self.y < (params.ymax + 0.25)) and (top_fade > 0.0)


class FormulaParticleHorizontal(FormulaParticleBase):
    __slots__ = ()

    def step(self, dt: float, params: EmissionParams) -> bool:
        self.age += dt
        t = self.age

        # rise
        self.y += self.vy * dt * params.speed_scale
        self.vy *= 0.996

        # wander increases with height
        height_gain = 0.6 + 1.2 * min(1.0, self.y / max(0.8, params.ymax))
        f = 0.9 * math.sin(self.seed + 1.6 * t + self.phase) + 0.7 * math.sin(self.seed * 0.5 + 2.7 * t)
        noise = random.uniform(-1.0, 1.0)

        turb = max(0.0, params.turbulence)
        # vx is an OU-ish velocity; stronger wander if user increases h_wander
        self.vx += turb * (0.40 * f + 0.25 * noise) * dt * height_gain * params.speed_scale
        self.vx *= 0.985
        self.x += self.vx * dt * params.speed_scale * max(0.25, params.h_wander) * SCALE_X

        top_fade = 1.0 - smoothstep(params.ymax - 0.35, params.ymax - 0.02, self.y)
        return (self.age < self.life) and (self.y < (params.ymax + 0.25)) and (top_fade > 0.0)


# -----------------------------
# Ember particles (kept)
# -----------------------------
class EmberParticle:
    __slots__ = ("x", "y", "vx", "vy", "life", "age", "size", "seed")

    def __init__(self, x0, y0, vx, vy, life, size):
        self.x = x0
        self.y = y0
        self.vx = vx
        self.vy = vy
        self.life = life
        self.age = 0.0
        self.size = size
        self.seed = random.uniform(0, 1000)

    def step(self, dt):
        self.age += dt
        wobble = 0.06 * math.sin(self.seed + self.age * 5.2)
        self.x += (self.vx + wobble) * dt
        self.y += self.vy * dt
        self.vy += EMBER_GRAVITY * dt
        return self.age < self.life

    @property
    def alpha(self):
        t = self.age / self.life
        fade_in = min(1.0, t / 0.08)
        fade_out = max(0.0, (1.0 - t) / 0.45)
        return min(fade_in, fade_out)

    @property
    def color(self):
        t = self.age / self.life
        flick = 0.85 + 0.15 * math.sin(self.seed + self.age * (2 * math.pi) * (EMBER_FLICKER / 10.0))
        if t < 0.25:
            u = t / 0.25
            r, g, b = 1.0, 1.0 - 0.15 * u, 1.0 - 0.65 * u
        elif t < 0.70:
            u = (t - 0.25) / 0.45
            r, g, b = 1.0, 0.85 - 0.65 * u, 0.35 - 0.35 * u
        else:
            u = (t - 0.70) / 0.30
            r, g, b = 1.0 - 0.55 * u, 0.20 - 0.20 * u, 0.05 - 0.05 * u
        r = max(0.0, min(1.0, r * flick))
        g = max(0.0, min(1.0, g * flick))
        b = max(0.0, min(1.0, b * flick))
        return (r, g, b)


# -----------------------------
# Build GIF (headless) with progress callback
# -----------------------------
def build_gif(
    formulas: List[str],
    params: EmissionParams,
    out_path: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Builds a GIF using the same rendering approach as the original script:
    render frames to PNG, then encode with Pillow.
    progress_cb(frame_index_1based, total_frames) is called during rendering.
    """
    global PLUME_AXIS_X

    # local particle state
    particles: List[FormulaParticleBase] = []
    text_artists: List[mpl.text.Text] = []
    embers: List[EmberParticle] = []

    def spawn_particles():
        n = int(params.emit_rate)
        if random.random() < (params.emit_rate - n):
            n += 1
        if random.random() < 0.20:
            n += 1
        for _ in range(n):
            if len(particles) >= params.max_particles:
                break
            text = random.choice(formulas) if formulas else "VOC"
            y0 = random.uniform(FIRE_Y1 - 0.02, FIRE_Y1 + 0.03)

            if params.mode == "cone":
                x0 = random.uniform(0.5 - 0.015 * SCALE_X, 0.5 + 0.015 * SCALE_X)
                vy = random.uniform(0.32, 0.48)
                life = random.uniform(6.5, 9.0)
                size = random.uniform(9, 14)
                particles.append(FormulaParticleCone(text, x0, y0, vy, life, size, params))
            else:
                x0 = random.uniform(params.h_xmin, params.h_xmax)
                x0 = XMIN + (x0 * (XMAX - XMIN))  # interpret h_xmin/h_xmax in 0..1 of scene width
                vy = random.uniform(0.34, 0.52)
                vx = random.uniform(-0.05, 0.05)
                life = random.uniform(6.5, 9.5)
                size = random.uniform(9, 14)
                particles.append(FormulaParticleHorizontal(text, x0, y0, vx=vx, vy=vy, life=life, size=size))

    def spawn_embers():
        if not ENABLE_EMBERS:
            return
        n = int(EMBER_RATE)
        if random.random() < (EMBER_RATE - n):
            n += 1
        for _ in range(n):
            if len(embers) >= MAX_EMBERS:
                break
            x0 = random.uniform(0.5 - 0.020 * SCALE_X, 0.5 + 0.020 * SCALE_X)
            y0 = random.uniform(FIRE_Y1 - 0.01, FIRE_Y1 + 0.03)
            vx = random.uniform(*EMBER_VX_RANGE) * params.speed_scale
            vy = random.uniform(*EMBER_VY_RANGE) * params.speed_scale
            life = random.uniform(*EMBER_LIFE_RANGE)
            size = random.uniform(*EMBER_SIZE_RANGE)
            embers.append(EmberParticle(x0, y0, vx, vy, life, size))

    def ensure_text_artists(ax):
        while len(text_artists) < len(particles):
            txt = ax.text(
                0, 0, "",
                color="white",
                fontsize=11,
                alpha=0.0,
                ha="center",
                va="center",
                zorder=10
            )
            txt.set_clip_on(True)
            txt.set_clip_path(ax.patch)
            text_artists.append(txt)

    # Figure setup (same style as original)
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, params.ymax)

    smoke_im = ax.imshow(
        np.zeros((H, W, 4), dtype=np.float32),
        extent=(XMIN, XMAX, 0, params.ymax),
        origin="lower",
        interpolation="bilinear",
        zorder=6
    )

    ember_sc = ax.scatter([], [], s=[], c=[], marker="o", linewidths=0, zorder=9)
    ember_sc.set_clip_on(True)
    ember_sc.set_clip_path(ax.patch)

    def update(frame_idx):
        nonlocal particles, embers
        t = frame_idx / FPS
        dt = 1.0 / FPS

        # plume axis wander for cone
        if params.mode == "cone":
            PLUME_AXIS_X = 0.50 + float(params.axis_wander) * math.sin(0.25 * t)
        else:
            PLUME_AXIS_X = 0.50

        # fire & smoke (kept: smoke disabled by default)
        f0 = np.zeros((H, W), dtype=np.float32)
        srgba = smoke_field(t, f0)
        smoke_im.set_data(srgba)

        spawn_particles()
        spawn_embers()

        alive = []
        for p in particles:
            if p.step(dt, params) and (XMIN <= p.x <= XMAX) and (0.0 <= p.y <= params.ymax):
                alive.append(p)
        particles = alive

        alive_e = []
        for e in embers:
            if e.step(dt) and (XMIN <= e.x <= XMAX) and (0.0 <= e.y <= params.ymax):
                alive_e.append(e)
        embers = alive_e

        if ENABLE_EMBERS:
            if embers:
                ember_sc.set_offsets([(e.x, e.y) for e in embers])
                ember_sc.set_sizes([e.size for e in embers])
                ember_sc.set_color([(*e.color, e.alpha) for e in embers])
            else:
                ember_sc.set_offsets([])
                ember_sc.set_sizes([])
                ember_sc.set_color([])

        ensure_text_artists(ax)

        for i, txt in enumerate(text_artists):
            if i < len(particles):
                p = particles[i]
                txt.set_text(p.text)
                txt.set_position((p.x, p.y))
                txt.set_alpha(p.alpha)
                txt.set_fontsize(p.size * (0.95 + 0.08 * math.sin(p.age * 2.6 + 0.4)))
                txt.set_color(text_color_for_height(p.y, p.y0, params.ymax, params.gradient))
            else:
                txt.set_alpha(0.0)

        return [smoke_im, ember_sc, *text_artists]

    anim = FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS, blit=False)

    # Save with solid white background, with progress callback
    out_path = Path(out_path).expanduser().resolve()
    frames: List[Image.Image] = []
    duration_ms = int(1000 / FPS)

    for i in range(FRAMES):
        anim._draw_next_frame(i, blit=False)  # type: ignore
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi, facecolor="white", edgecolor="white", transparent=False, pad_inches=0)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        im = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.NONE)
        frames.append(im)
        if progress_cb is not None:
            progress_cb(i + 1, FRAMES)

    if not frames:
        plt.close(fig)
        raise RuntimeError("No frames were rendered; GIF not created.")

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    plt.close(fig)


# -----------------------------
# PyQt GUI
# -----------------------------
def _qt_imports():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets
    except Exception:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets


def run_gui() -> int:
    QtCore, QtGui, QtWidgets = _qt_imports()

    class PasteTable(QtWidgets.QTableWidget):
        """A 1-column table that supports Excel-style paste (one item per row)."""
        def __init__(self, parent=None):
            super().__init__(0, 1, parent)
            self.setHorizontalHeaderLabels(["Compound / Formula"])
            self.horizontalHeader().setStretchLastSection(True)
            self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
            self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        def keyPressEvent(self, event):  # type: ignore
            if event.matches(QtGui.QKeySequence.StandardKey.Paste):  # type: ignore
                text = QtWidgets.QApplication.clipboard().text()
                self.paste_text(text)
                return
            super().keyPressEvent(event)

        def paste_text(self, text: str):
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                return
            # start at current row or append
            row = self.currentRow()
            if row < 0:
                row = self.rowCount()
            needed = row + len(lines)
            if self.rowCount() < needed:
                self.setRowCount(needed)
            for i, ln in enumerate(lines):
                self.setItem(row + i, 0, QtWidgets.QTableWidgetItem(ln))

        def get_values(self) -> List[str]:
            vals: List[str] = []
            for r in range(self.rowCount()):
                it = self.item(r, 0)
                if it is None:
                    continue
                v = it.text().strip()
                if v:
                    vals.append(v)
            return vals

        def set_values(self, values: List[str]):
            self.setRowCount(len(values))
            for r, v in enumerate(values):
                self.setItem(r, 0, QtWidgets.QTableWidgetItem(str(v)))

    class Worker(QtCore.QThread):  # type: ignore
        progress = QtCore.pyqtSignal(int)  # type: ignore
        status = QtCore.pyqtSignal(str)    # type: ignore
        done = QtCore.pyqtSignal(str)      # type: ignore
        failed = QtCore.pyqtSignal(str)    # type: ignore

        def __init__(self, formulas: List[str], params: EmissionParams, out_path: Path):
            super().__init__()
            self.formulas = formulas
            self.params = params
            self.out_path = out_path

        def run(self):  # type: ignore
            try:
                def cb(i: int, n: int):
                    pct = int(round(100 * (i / float(n))))
                    self.progress.emit(pct)

                self.status.emit("Rendering frames…")
                build_gif(self.formulas, self.params, self.out_path, progress_cb=cb)
                self.progress.emit(100)
                self.done.emit(str(self.out_path))
            except Exception as e:
                self.failed.emit(str(e))

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Fire formula GIF builder")

            # scrollable central
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            central = QtWidgets.QWidget()
            scroll.setWidget(central)
            self.setCentralWidget(scroll)
            layout = QtWidgets.QVBoxLayout(central)

            # formulas table
            box_form = QtWidgets.QGroupBox("Compounds (paste one per line)")
            v = QtWidgets.QVBoxLayout(box_form)
            self.table = PasteTable()
            self.table.setMinimumHeight(220)
            v.addWidget(self.table)
            btns = QtWidgets.QHBoxLayout()
            self.btn_add_row = QtWidgets.QPushButton("Add row")
            self.btn_clear = QtWidgets.QPushButton("Clear")
            self.btn_load_defaults = QtWidgets.QPushButton("Load defaults")
            btns.addWidget(self.btn_add_row)
            btns.addWidget(self.btn_clear)
            btns.addWidget(self.btn_load_defaults)
            btns.addStretch(1)
            v.addLayout(btns)
            layout.addWidget(box_form)

            # emission settings
            box_emit = QtWidgets.QGroupBox("Emission settings")
            form = QtWidgets.QFormLayout(box_emit)

            self.combo_mode = QtWidgets.QComboBox()
            self.combo_mode.addItems(["cone", "horizontal"])
            form.addRow("Emission space:", self.combo_mode)

            self.spin_ymax = QtWidgets.QDoubleSpinBox()
            self.spin_ymax.setRange(1.0, 8.0)
            self.spin_ymax.setValue(YMAX_DEFAULT)
            self.spin_ymax.setDecimals(2)
            form.addRow("Rise height (YMAX):", self.spin_ymax)

            self.spin_emit_rate = QtWidgets.QDoubleSpinBox()
            self.spin_emit_rate.setRange(0.01, 10.0)
            self.spin_emit_rate.setSingleStep(0.05)
            self.spin_emit_rate.setValue(0.30)
            form.addRow("Emit rate (avg per frame):", self.spin_emit_rate)

            self.spin_max_particles = QtWidgets.QSpinBox()
            self.spin_max_particles.setRange(10, 2000)
            self.spin_max_particles.setValue(240)
            form.addRow("Max particles:", self.spin_max_particles)

            self.spin_speed = QtWidgets.QDoubleSpinBox()
            self.spin_speed.setRange(0.05, 3.0)
            self.spin_speed.setSingleStep(0.05)
            self.spin_speed.setValue(0.80)
            form.addRow("Movement speed:", self.spin_speed)

            self.spin_turb = QtWidgets.QDoubleSpinBox()
            self.spin_turb.setRange(0.0, 5.0)
            self.spin_turb.setSingleStep(0.1)
            self.spin_turb.setValue(1.0)
            form.addRow("Turbulence level:", self.spin_turb)

            self.combo_grad = QtWidgets.QComboBox()
            self.combo_grad.addItems(GRADIENT_PRESETS)
            self.combo_grad.setCurrentText(GRADIENT_PRESETS[0])
            form.addRow("Color gradient:", self.combo_grad)

            # cone controls
            self.spin_cone_base = QtWidgets.QDoubleSpinBox()
            self.spin_cone_base.setRange(0.0, 0.30)
            self.spin_cone_base.setDecimals(3)
            self.spin_cone_base.setValue(0.030)
            form.addRow("Cone base half-width:", self.spin_cone_base)

            self.spin_cone_k = QtWidgets.QDoubleSpinBox()
            self.spin_cone_k.setRange(0.0, 5.0)
            self.spin_cone_k.setDecimals(2)
            self.spin_cone_k.setValue(1.20)
            form.addRow("Cone expansion (k):", self.spin_cone_k)

            self.spin_axis_wander = QtWidgets.QDoubleSpinBox()
            self.spin_axis_wander.setRange(0.0, 0.20)
            self.spin_axis_wander.setDecimals(3)
            self.spin_axis_wander.setValue(0.030)
            form.addRow("Cone axis wander:", self.spin_axis_wander)

            # horizontal controls
            self.spin_hxmin = QtWidgets.QDoubleSpinBox()
            self.spin_hxmin.setRange(0.0, 1.0)
            self.spin_hxmin.setDecimals(3)
            self.spin_hxmin.setValue(0.40)
            form.addRow("Horizontal X min (0..1):", self.spin_hxmin)

            self.spin_hxmax = QtWidgets.QDoubleSpinBox()
            self.spin_hxmax.setRange(0.0, 1.0)
            self.spin_hxmax.setDecimals(3)
            self.spin_hxmax.setValue(0.60)
            form.addRow("Horizontal X max (0..1):", self.spin_hxmax)

            self.spin_hwander = QtWidgets.QDoubleSpinBox()
            self.spin_hwander.setRange(0.0, 2.0)
            self.spin_hwander.setDecimals(2)
            self.spin_hwander.setValue(0.10)
            form.addRow("Horizontal wander:", self.spin_hwander)

            layout.addWidget(box_emit)

            # output + build
            box_out = QtWidgets.QGroupBox("Output")
            f2 = QtWidgets.QFormLayout(box_out)
            self.ed_out = QtWidgets.QLineEdit(str(Path.cwd() / DEFAULT_GIF_NAME))
            self.btn_browse = QtWidgets.QPushButton("Browse…")
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.ed_out, 1)
            row.addWidget(self.btn_browse)
            f2.addRow("GIF file:", row)

            self.btn_build = QtWidgets.QPushButton("Build GIF")
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.lbl_status = QtWidgets.QLabel("")
            f2.addRow(self.btn_build)
            f2.addRow("Progress:", self.progress)
            f2.addRow("Status:", self.lbl_status)

            layout.addWidget(box_out)

            self.worker: Optional[Worker] = None

            # wiring
            self.btn_add_row.clicked.connect(self.add_row)
            self.btn_clear.clicked.connect(self.clear_table)
            self.btn_load_defaults.clicked.connect(self.load_defaults)
            self.btn_browse.clicked.connect(self.browse_out)
            self.btn_build.clicked.connect(self.build)

            self.combo_mode.currentTextChanged.connect(self._sync_mode_controls)
            self._sync_mode_controls(self.combo_mode.currentText())

            # seed defaults
            self.load_defaults()

        def _sync_mode_controls(self, mode: str):
            is_cone = (mode == "cone")
            for w in [self.spin_cone_base, self.spin_cone_k, self.spin_axis_wander]:
                w.setEnabled(is_cone)
            for w in [self.spin_hxmin, self.spin_hxmax, self.spin_hwander]:
                w.setEnabled(not is_cone)

        def add_row(self):
            r = self.table.rowCount()
            self.table.setRowCount(r + 1)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(""))

        def clear_table(self):
            self.table.setRowCount(0)

        def load_defaults(self):
            defaults = [
                "VOCs", "IVOCs", "SVOCs",
                "HCHO", "HCl",
                "C10H16", "C10H16O2", "C10H8", "C8H6O4",
                "C8H14N2", "C9H16N2", "C6H10O5", "C8H12O5",
                "CH3COOH", "C7H10O4"
            ]
            self.table.set_values(defaults)

        def browse_out(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save GIF", self.ed_out.text(), "GIF (*.gif)")
            if path:
                if not path.lower().endswith(".gif"):
                    path += ".gif"
                self.ed_out.setText(path)

        def _collect_params(self) -> EmissionParams:
            mode = str(self.combo_mode.currentText())
            p = EmissionParams(
                mode=mode,
                emit_rate=float(self.spin_emit_rate.value()),
                max_particles=int(self.spin_max_particles.value()),
                speed_scale=float(self.spin_speed.value()),
                turbulence=float(self.spin_turb.value()),
                gradient=str(self.combo_grad.currentText()),
                ymax=float(self.spin_ymax.value()),
                cone_base_w=float(self.spin_cone_base.value()),
                cone_k=float(self.spin_cone_k.value()),
                axis_wander=float(self.spin_axis_wander.value()),
                h_xmin=float(self.spin_hxmin.value()),
                h_xmax=float(self.spin_hxmax.value()),
                h_wander=float(self.spin_hwander.value()),
            )
            # sanity for horizontal width
            if p.h_xmax < p.h_xmin:
                p.h_xmin, p.h_xmax = p.h_xmax, p.h_xmin
            return p

        def build(self):
            formulas = self.table.get_values()
            if not formulas:
                QtWidgets.QMessageBox.warning(self, "Missing compounds", "Paste or enter at least one compound/formula.")
                return

            out = Path(self.ed_out.text().strip() or DEFAULT_GIF_NAME).expanduser().resolve()
            params = self._collect_params()

            self.progress.setValue(0)
            self.lbl_status.setText("Starting…")
            self.btn_build.setEnabled(False)

            self.worker = Worker(formulas, params, out)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.status.connect(self.lbl_status.setText)
            self.worker.done.connect(self._on_done)
            self.worker.failed.connect(self._on_failed)
            self.worker.start()

        def _on_done(self, out_path: str):
            self.btn_build.setEnabled(True)
            self.lbl_status.setText("Done")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved GIF:\n{out_path}")

        def _on_failed(self, err: str):
            self.btn_build.setEnabled(True)
            self.lbl_status.setText("Failed")
            QtWidgets.QMessageBox.critical(self, "Build failed", err)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(980, 900)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_gui())
