"""
revealtimeseriesgif.py

Reveal-animation GIF from CSV / Excel / TXT with a PyQt GUI.

Adds:
- Multiple timestamp-set X-axis handling in overplot mode (top/bottom/multiple axes)
- Axis/text/legend color + font controls
- Larger column-selection area
- Live preview of the fully-revealed plot that updates as you change settings
- GIF loop toggle (loop forever vs "run once-ish")

Run
  python revealtimeseriesgif3_updated.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


# ----------------------------
# Data loading / parsing
# ----------------------------


def sniff_delimiter(sample: str) -> str:
    candidates = [",", ";", "\t", "|"]
    best = ","
    best_score = -1
    lines = [ln for ln in sample.splitlines() if ln.strip()][:40]
    for d in candidates:
        counts = [len(ln.split(d)) for ln in lines]
        if len(counts) < 3:
            continue
        score = (np.median(counts), -np.std(counts))
        scalar = float(score[0]) + float(score[1]) * 0.01
        if scalar > best_score:
            best_score = scalar
            best = d
    return best


def load_table(path: Path, sheet: Optional[str] = None, delimiter: Optional[str] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet if sheet else 0, header=0)

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if delimiter is None:
        delimiter = sniff_delimiter(raw[:8000])

    return pd.read_csv(path, sep=delimiter, header=0)


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_x_series(df: pd.DataFrame, x_col: str, time_mode: str) -> Tuple[np.ndarray, Optional[pd.Series]]:
    """
    Return x as numeric array for plotting, plus optional datetime Series.

    - datetime: x is Matplotlib date numbers (days-as-float), so axes can be formatted as dates
    - numeric: x is numeric shifted to start at 0
    - auto: try both; pick the one with more usable values (ties prefer datetime if viable)
    """
    x_dt: Optional[pd.Series] = None
    x_mpl: Optional[np.ndarray] = None
    ok_dt = 0

    if time_mode in ("auto", "datetime"):
        x_dt = pd.to_datetime(df[x_col], errors="coerce", utc=False)
        ok_dt = int(x_dt.notna().sum())
        if ok_dt > 0:
            dt_naive = x_dt.dt.tz_localize(None) if hasattr(x_dt.dt, "tz_localize") else x_dt
            x_mpl = mdates.date2num(dt_naive.to_numpy())
        if time_mode == "datetime":
            return (x_mpl if x_mpl is not None else np.full(len(df), np.nan, dtype=float)), x_dt

    x_num = coerce_numeric_series(df[x_col]).to_numpy(dtype=float, copy=False)
    ok_num = int(np.isfinite(x_num).sum())
    if ok_num > 0:
        x_num = x_num - float(np.nanmin(x_num[np.isfinite(x_num)]))

    if time_mode == "auto" and x_mpl is not None:
        ok_mpl = int(np.isfinite(x_mpl).sum())
        if ok_mpl >= max(3, ok_num):
            return x_mpl, x_dt

    return x_num, None


# ----------------------------
# Specs
# ----------------------------


@dataclass
class SeriesSpec:
    label: str
    x: np.ndarray
    y: np.ndarray
    set_id: int = 1
    is_datetime: bool = False
    legend_label: Optional[str] = None

    plot_type: str = "markers"  # bar | markers | line | line+markers
    color: Optional[str] = None
    marker: str = "o"
    marker_size: float = 20.0
    line_width: float = 2.0
    line_style: str = "-"
    alpha: float = 1.0


@dataclass
class LegendSpec:
    show: bool = True
    loc: str = "best"
    title: str = ""
    fontsize: int = 10
    title_fontsize: int = 10
    frame: bool = True
    facecolor: str = "white"
    edgecolor: str = "black"
    framealpha: float = 0.9
    textcolor: str = "black"


@dataclass
class StyleSpec:
    font_family: str = "DejaVu Sans"
    base_font_size: int = 10
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    background: str = "white"  # "white" or "none" for transparent
    grid_alpha: float = 0.25

    text_color: str = "black"
    title_color: str = "black"
    label_color: str = "black"
    tick_color: str = "black"
    spine_color: str = "black"


# ----------------------------
# Plot helpers
# ----------------------------


def _clean_and_sort(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if len(x2) < 2:
        raise ValueError("Not enough usable points after cleaning (NaNs/non-numeric).")
    order = np.argsort(x2)
    return x2[order], y2[order]


def _default_color_cycle(n: int) -> List[str]:
    prop = plt.rcParams.get("axes.prop_cycle")
    colors = prop.by_key().get("color", []) if prop else []
    if not colors:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    return [colors[i % len(colors)] for i in range(n)]


def _format_datetime_axis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _apply_axis_colors(ax, style: StyleSpec, x_color: Optional[str] = None):
    for side in ["bottom", "top", "left", "right"]:
        if side in ax.spines:
            ax.spines[side].set_color(style.spine_color)

    ax.tick_params(axis="y", colors=style.tick_color)
    ax.yaxis.label.set_color(style.label_color)

    if x_color:
        ax.tick_params(axis="x", colors=x_color)
        ax.xaxis.label.set_color(x_color)
        if "bottom" in ax.spines:
            ax.spines["bottom"].set_color(x_color)
        if "top" in ax.spines:
            ax.spines["top"].set_color(x_color)
    else:
        ax.tick_params(axis="x", colors=style.tick_color)
        ax.xaxis.label.set_color(style.label_color)


def _make_overplot_axes(fig, set_ids: List[int], xaxis_mode: str) -> Dict[int, mpl.axes.Axes]:
    """
    Returns mapping set_id -> axis. Base axis is always set_ids[0].
    xaxis_mode:
      - single_bottom / single_top: one axis only
      - multi_top: base bottom + others stacked on top
      - multi_bottom: base bottom + others stacked below
      - first_bottom_others_top: base bottom + others on top
    """
    base_ax = fig.add_subplot(111)
    setattr(base_ax, '_ts_set_id', set_ids[0])
    mapping: Dict[int, mpl.axes.Axes] = {set_ids[0]: base_ax}

    if len(set_ids) <= 1 or xaxis_mode in ("single_bottom", "single_top"):
        if xaxis_mode == "single_top":
            base_ax.xaxis.set_label_position("top")
            base_ax.xaxis.tick_top()
        return mapping

    for j, sid in enumerate(set_ids[1:], start=1):
        ax2 = base_ax.twiny()
        if xaxis_mode in ("multi_top", "first_bottom_others_top"):
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()
            ax2.spines["top"].set_position(("outward", 18 * (j - 1)))
        elif xaxis_mode == "multi_bottom":
            ax2.xaxis.set_label_position("bottom")
            ax2.xaxis.tick_bottom()
            ax2.spines["bottom"].set_position(("outward", 18 * j))
            ax2.spines["top"].set_visible(False)
        else:
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()
        ax2.grid(False)
        mapping[sid] = ax2
        setattr(ax2, '_ts_set_id', sid)

    return mapping


def draw_full_reveal_preview(
    fig: "plt.Figure",
    series: List[SeriesSpec],
    mode: str,
    title: str,
    xlabel: str,
    ylabel: str,
    style: StyleSpec,
    legend: LegendSpec,
    xaxis_mode: str,
    xaxis_colors: Optional[Dict[int, str]] = None,
) -> None:
    if xaxis_colors is None:
        xaxis_colors = {}

    fig.clear()
    if title:
        fig.suptitle(title, color=style.title_color)

    if not series:
        fig.canvas.draw_idle()
        return

    # fill colors
    cyc = _default_color_cycle(len(series))
    for i, s in enumerate(series):
        if not s.color:
            s.color = cyc[i]
        if not s.legend_label:
            s.legend_label = s.label

    if mode == "stacked":
        axes = fig.subplots(len(series), 1, sharex=False)
        if len(series) == 1:
            axes = [axes]
        axes = list(axes)
        for ax, s in zip(axes, series):
            ax.set_facecolor(style.background)
            ax.grid(False, alpha=style.grid_alpha)
            _apply_axis_colors(ax, style, xaxis_colors.get(s.set_id))
            if s.is_datetime:
                _format_datetime_axis(ax)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(str(s.legend_label or s.label))

            if s.plot_type == "line":
                ax.plot(s.x, s.y, color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha)
            elif s.plot_type == "markers":
                ax.scatter(s.x, s.y, s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha)
            elif s.plot_type == "line+markers":
                ax.plot(s.x, s.y, color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha)
                ax.scatter(s.x, s.y, s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha)
            elif s.plot_type == "bar":
                ax.bar(s.x, s.y, color=s.color, alpha=s.alpha)

        fig.tight_layout()
        return

    # overplot
    set_ids = sorted({s.set_id for s in series})
    ax_map = _make_overplot_axes(fig, set_ids, xaxis_mode)
    base_ax = ax_map[set_ids[0]]

    # global y
    y_all = np.concatenate([s.y[np.isfinite(s.y)] for s in series])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
    for ax in set(ax_map.values()):
        ax.set_facecolor(style.background)
        ax.grid(False, alpha=style.grid_alpha)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    base_ax.set_ylabel(ylabel)

    # x limits + labels per axis
    for sid, ax in ax_map.items():
        _apply_axis_colors(ax, style, xaxis_colors.get(sid))
        ax.set_xlabel(xlabel)

        x_all = np.concatenate([s.x[np.isfinite(s.x)] for s in series if s.set_id == sid])
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        if all(s.is_datetime for s in series if s.set_id == sid):
            _format_datetime_axis(ax)

    # plot full
    for s in series:
        ax = ax_map.get(s.set_id, base_ax)
        if s.plot_type == "line":
            ax.plot(s.x, s.y, color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha, label=s.legend_label)
        elif s.plot_type == "markers":
            ax.scatter(s.x, s.y, s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha, label=s.legend_label)
        elif s.plot_type == "line+markers":
            ax.plot(s.x, s.y, color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha, label=s.legend_label)
            ax.scatter(s.x, s.y, s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha)
        elif s.plot_type == "bar":
            ax.bar(s.x, s.y, color=s.color, alpha=s.alpha, label=s.legend_label)

    if legend.show:
        leg = base_ax.legend(
            loc=legend.loc,
            title=legend.title if legend.title else None,
            frameon=legend.frame,
            fontsize=legend.fontsize,
            title_fontsize=legend.title_fontsize,
        )
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor(legend.facecolor)
            frame.set_edgecolor(legend.edgecolor)
            frame.set_alpha(legend.framealpha)
            for t in leg.get_texts():
                t.set_color(legend.textcolor)
            if leg.get_title() is not None:
                leg.get_title().set_color(legend.textcolor)

    fig.tight_layout()


# ----------------------------
# GIF building
# ----------------------------


def build_reveal_markers_gif_multi(
    series: List[SeriesSpec],
    out_path: Path,
    mode: str = "overplot",
    fps: int = 30,
    duration: float = 12.0,
    dpi: int = 120,
    figsize: Tuple[float, float] = (1280 / 120, 720 / 120),
    title: str = "",
    xlabel: str = "Time",
    ylabel: str = "Value",
    style: Optional[StyleSpec] = None,
    legend: Optional[LegendSpec] = None,
    xaxis_mode: str = "single_bottom",
    xaxis_colors: Optional[Dict[int, str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    loop_forever: bool = True,
) -> None:
    if style is None:
        style = StyleSpec()
    if legend is None:
        legend = LegendSpec()
    if xaxis_colors is None:
        xaxis_colors = {}

    if not series:
        raise ValueError("No series selected.")

    cleaned: List[SeriesSpec] = []
    for s in series:
        x2, y2 = _clean_and_sort(s.x, s.y)
        cleaned.append(
            SeriesSpec(
                label=s.label,
                x=x2,
                y=y2,
                set_id=s.set_id,
                is_datetime=s.is_datetime,
                legend_label=s.legend_label or s.label,
                plot_type=s.plot_type,
                color=s.color,
                marker=s.marker,
                marker_size=s.marker_size,
                line_width=s.line_width,
                line_style=s.line_style,
                alpha=s.alpha,
            )
        )

    cyc = _default_color_cycle(len(cleaned))
    for i, s in enumerate(cleaned):
        if not s.color:
            s.color = cyc[i]

    n_frames = max(2, int(round(fps * duration)))
    idx_per_frame = [np.linspace(1, len(s.x), n_frames).astype(int) for s in cleaned]

    rc = {
        "font.family": style.font_family,
        "font.size": style.base_font_size,
        "axes.titlesize": style.title_size,
        "axes.labelsize": style.label_size,
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "text.color": style.text_color,
        "axes.labelcolor": style.label_color,
        "xtick.color": style.tick_color,
        "ytick.color": style.tick_color,
        "axes.edgecolor": style.spine_color,
    }

    with mpl.rc_context(rc=rc):
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=style.background)
        if title:
            fig.suptitle(title, color=style.title_color)

        artists: List[Dict[str, object]] = []
        leaders: List[Dict[str, object]] = []  # moving 'head' markers/highlights

        def add_series(ax, s: SeriesSpec) -> Dict[str, object]:
            if s.plot_type == "line":
                ln, = ax.plot([], [], color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha, label=s.legend_label)
                return {"kind": "line", "artist": ln}
            if s.plot_type == "markers":
                sc = ax.scatter([], [], s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha, label=s.legend_label)
                return {"kind": "scatter", "artist": sc}
            if s.plot_type == "line+markers":
                ln, = ax.plot([], [], color=s.color, linewidth=s.line_width, linestyle=s.line_style, alpha=s.alpha, label=s.legend_label)
                sc = ax.scatter([], [], s=s.marker_size, marker=s.marker, color=s.color, alpha=s.alpha)
                return {"kind": "line+markers", "line": ln, "scatter": sc}
            if s.plot_type == "bar":
                bars = ax.bar(s.x, np.zeros_like(s.y), color=s.color, alpha=s.alpha, label=s.legend_label)
                return {"kind": "bar", "bars": bars}
            raise ValueError(f"Unknown plot_type: {s.plot_type}")

        if mode == "stacked":
            axes = fig.subplots(len(cleaned), 1, sharex=False)
            if len(cleaned) == 1:
                axes = [axes]
            axes = list(axes)

            for ax, s in zip(axes, cleaned):
                ax.set_facecolor(style.background)
                ax.grid(False, alpha=style.grid_alpha)
                _apply_axis_colors(ax, style, xaxis_colors.get(s.set_id))
                if s.is_datetime:
                    _format_datetime_axis(ax)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(str(s.legend_label or s.label))

                # limits
                x_all = s.x[np.isfinite(s.x)]
                x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
                x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
                ax.set_xlim(x_min - x_pad, x_max + x_pad)

                y_min, y_max = float(np.min(s.y)), float(np.max(s.y))
                y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

                artists.append(add_series(ax, s))
                # Leader point/highlight
                if s.plot_type in ("line", "markers", "line+markers"):
                    lead_size = float(max(80.0, s.marker_size * 3.0))
                    lead = ax.scatter([], [], s=lead_size, marker=(s.marker or "o"),
                                      color=s.color, alpha=1.0,
                                      edgecolors="white", linewidths=1.2, zorder=10)
                    leaders.append({"kind": "point", "artist": lead})
                else:
                    # bar: we'll highlight the active bar via edgecolor/linewidth
                    leaders.append({"kind": "bar", "bars": artists[-1].get("bars")})

        else:
            set_ids = sorted({s.set_id for s in cleaned})
            ax_map = _make_overplot_axes(fig, set_ids, xaxis_mode)
            base_ax = ax_map[set_ids[0]]

            # global y
            y_all = np.concatenate([s.y[np.isfinite(s.y)] for s in cleaned])
            y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
            y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
            for ax in set(ax_map.values()):
                ax.set_facecolor(style.background)
                ax.grid(False, alpha=style.grid_alpha)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            base_ax.set_ylabel(ylabel)

            # per-axis x limits/format/colors
            for sid, ax in ax_map.items():
                _apply_axis_colors(ax, style, xaxis_colors.get(sid))
                ax.set_xlabel(xlabel)
                x_all = np.concatenate([s.x[np.isfinite(s.x)] for s in cleaned if s.set_id == sid])
                x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
                x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                if all(s.is_datetime for s in cleaned if s.set_id == sid):
                    _format_datetime_axis(ax)

            for s in cleaned:
                ax_used = ax_map.get(s.set_id, base_ax)
                artists.append(add_series(ax_used, s))
                # Leader point/highlight
                if s.plot_type in ("line", "markers", "line+markers"):
                    lead_size = float(max(80.0, s.marker_size * 3.0))
                    lead = ax_used.scatter([], [], s=lead_size, marker=(s.marker or "o"),
                                           color=s.color, alpha=1.0,
                                           edgecolors="white", linewidths=1.2, zorder=10)
                    leaders.append({"kind": "point", "artist": lead})
                else:
                    leaders.append({"kind": "bar", "bars": artists[-1].get("bars")})

            # legend
            if legend.show:
                handles: List[Line2D] = []
                labels: List[str] = []
                for s, entry in zip(cleaned, artists):
                    if entry["kind"] == "scatter":
                        handles.append(Line2D([0], [0], linestyle="", marker=s.marker, color=s.color))
                    elif entry["kind"] == "bar":
                        handles.append(Line2D([0], [0], linewidth=6, color=s.color))
                    elif entry["kind"] == "line+markers":
                        handles.append(entry["line"])  # type: ignore
                    else:
                        handles.append(entry.get("artist"))  # type: ignore
                    labels.append(str(s.legend_label or s.label))

                leg = base_ax.legend(
                    handles,
                    labels,
                    loc=legend.loc,
                    title=legend.title if legend.title else None,
                    frameon=legend.frame,
                    fontsize=legend.fontsize,
                    title_fontsize=legend.title_fontsize,
                )
                if leg is not None:
                    frame = leg.get_frame()
                    frame.set_facecolor(legend.facecolor)
                    frame.set_edgecolor(legend.edgecolor)
                    frame.set_alpha(legend.framealpha)
                    for t in leg.get_texts():
                        t.set_color(legend.textcolor)
                    if leg.get_title() is not None:
                        leg.get_title().set_color(legend.textcolor)

        def init():
            blit_artists: List[object] = []
            # reset main artists
            for entry in artists:
                kind = entry["kind"]
                if kind == "line":
                    ln: Line2D = entry["artist"]  # type: ignore
                    ln.set_data([], [])
                    blit_artists.append(ln)
                elif kind == "scatter":
                    sc = entry["artist"]  # type: ignore
                    sc.set_offsets(np.empty((0, 2)))
                    blit_artists.append(sc)
                elif kind == "line+markers":
                    ln: Line2D = entry["line"]  # type: ignore
                    sc = entry["scatter"]  # type: ignore
                    ln.set_data([], [])
                    sc.set_offsets(np.empty((0, 2)))
                    blit_artists.extend([ln, sc])
                elif kind == "bar":
                    bars = entry["bars"]  # type: ignore
                    for r in bars:
                        r.set_height(0.0)
                        try:
                            r.set_edgecolor("none")
                            r.set_linewidth(0.0)
                        except Exception:
                            pass
                    blit_artists.extend(list(bars))

            # reset leader markers/highlights
            for l in leaders:
                if l.get("kind") == "point":
                    l["artist"].set_offsets(np.empty((0, 2)))  # type: ignore
                    blit_artists.append(l["artist"])  # type: ignore
                elif l.get("kind") == "bar":
                    bars = l.get("bars")
                    if bars is not None:
                        for r in bars:
                            try:
                                r.set_edgecolor("none")
                                r.set_linewidth(0.0)
                            except Exception:
                                pass
                        blit_artists.extend(list(bars))

            return tuple(blit_artists)

        def update(i: int):
            blit_artists: List[object] = []
            for entry, s, idx, lead in zip(artists, cleaned, idx_per_frame, leaders):
                k = int(idx[i])
                kind = entry["kind"]

                if kind == "line":
                    ln: Line2D = entry["artist"]  # type: ignore
                    ln.set_data(s.x[:k], s.y[:k])
                    blit_artists.append(ln)

                elif kind == "scatter":
                    sc = entry["artist"]  # type: ignore
                    sc.set_offsets(np.column_stack([s.x[:k], s.y[:k]]))
                    blit_artists.append(sc)

                elif kind == "line+markers":
                    ln: Line2D = entry["line"]  # type: ignore
                    sc = entry["scatter"]  # type: ignore
                    ln.set_data(s.x[:k], s.y[:k])
                    sc.set_offsets(np.column_stack([s.x[:k], s.y[:k]]))
                    blit_artists.extend([ln, sc])

                elif kind == "bar":
                    bars = entry["bars"]  # type: ignore
                    for j, r in enumerate(bars):
                        r.set_height(float(s.y[j]) if j < k else 0.0)
                    blit_artists.extend(list(bars))

                # leader point / highlight (once)
                if lead.get("kind") == "point":
                    if k > 0:
                        lead["artist"].set_offsets([[float(s.x[k - 1]), float(s.y[k - 1])]])  # type: ignore
                    else:
                        lead["artist"].set_offsets(np.empty((0, 2)))  # type: ignore
                    blit_artists.append(lead["artist"])  # type: ignore

                elif lead.get("kind") == "bar" and kind == "bar":
                    bars = entry["bars"]  # type: ignore
                    kk = max(0, k - 1)
                    for j, r in enumerate(bars):
                        try:
                            if j == kk:
                                r.set_edgecolor("white")
                                r.set_linewidth(2.0)
                            else:
                                r.set_edgecolor("none")
                                r.set_linewidth(0.0)
                        except Exception:
                            pass
                    blit_artists.extend(list(bars))

            return tuple(blit_artists)
        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)

        save_kwargs = {"writer": writer, "savefig_kwargs": {"facecolor": style.background}}
        try:
            anim.save(str(out_path), progress_callback=progress_callback, **save_kwargs)  # type: ignore
        except TypeError:
            anim.save(str(out_path), **save_kwargs)
        finally:
            plt.close(fig)

    # post-process loop flag
    try:
        from PIL import Image

        im = Image.open(out_path)
        frames = []
        try:
            while True:
                frames.append(im.copy())
                im.seek(im.tell() + 1)
        except EOFError:
            pass

        if frames:
            loop_val = 0 if loop_forever else 1
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                loop=loop_val,
                duration=int(round(1000 / fps)),
                disposal=2,
            )
    except Exception:
        pass


# ----------------------------
# PyQt GUI
# ----------------------------


def _qt_imports():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets
    except Exception:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets



def run_gui() -> int:
    QtCore, QtGui, QtWidgets = _qt_imports()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
    from matplotlib.figure import Figure  # type: ignore

    PLOT_TYPES = ["markers", "line", "line+markers", "bar"]
    MARKERS = ["o", ".", "x", "+", "s", "^", "v", "D", "*", "P", "X"]

    def _qt_match_exact_flag():
        try:
            return QtCore.Qt.MatchFlag.MatchExactly
        except Exception:
            return QtCore.Qt.MatchExactly

    def _qt_sel_mode_extended():
        try:
            return QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        except Exception:
            return QtWidgets.QAbstractItemView.ExtendedSelection

    class TimestampRow(QtWidgets.QGroupBox):
        changed = QtCore.pyqtSignal()  # type: ignore

        def __init__(self, title: str, parent=None):
            super().__init__(title, parent)
            self.x_combo = QtWidgets.QComboBox()
            self.y_list = QtWidgets.QListWidget()
            self.y_list.setSelectionMode(_qt_sel_mode_extended())
            self.y_list.setMinimumHeight(220)  # a bit taller for column selection
            self._build()

        def _build(self):
            form = QtWidgets.QFormLayout()
            form.addRow("Time column (x):", self.x_combo)
            form.addRow("Data columns (y):", self.y_list)
            self.setLayout(form)
            self.x_combo.currentIndexChanged.connect(self.changed.emit)
            self.y_list.itemSelectionChanged.connect(self.changed.emit)

        def set_columns(self, columns: List[str]):
            self.x_combo.clear()
            self.x_combo.addItems(columns)
            self.y_list.clear()
            for c in columns:
                self.y_list.addItem(str(c))

        def get_selection(self) -> Tuple[str, List[str]]:
            x_col = str(self.x_combo.currentText()).strip()
            y_cols = [i.text() for i in self.y_list.selectedItems()]
            return x_col, y_cols

    class LegendEditDialog(QtWidgets.QDialog):
        """Edits legend look + series labels. Writes back to the hidden legend widgets + styles table."""

        def __init__(self, parent: "MainWindow"):
            super().__init__(parent)
            self.setWindowTitle("Edit legend")
            self.parent = parent
            self.setModal(True)

            self.chk_show = QtWidgets.QCheckBox("Show legend")
            self.ed_title = QtWidgets.QLineEdit()
            self.combo_loc = QtWidgets.QComboBox()
            self.combo_loc.addItems([
                "best","upper right","upper left","lower left","lower right","center right","center left",
                "upper center","lower center","center"
            ])
            self.spin_font = QtWidgets.QSpinBox(); self.spin_font.setRange(6, 72)
            self.spin_title_font = QtWidgets.QSpinBox(); self.spin_title_font.setRange(6, 72)
            self.spin_alpha = QtWidgets.QDoubleSpinBox(); self.spin_alpha.setRange(0.0, 1.0); self.spin_alpha.setSingleStep(0.05)

            self.btn_face = QtWidgets.QPushButton("Pick…")
            self.btn_edge = QtWidgets.QPushButton("Pick…")
            self.btn_text = QtWidgets.QPushButton("Pick…")

            self.labels_table = QtWidgets.QTableWidget(0, 2)
            self.labels_table.setHorizontalHeaderLabels(["Series", "Legend label"])
            self.labels_table.horizontalHeader().setStretchLastSection(True)

            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)

            form = QtWidgets.QFormLayout()
            form.addRow("", self.chk_show)
            form.addRow("Legend title:", self.ed_title)
            form.addRow("Location:", self.combo_loc)
            fs = QtWidgets.QHBoxLayout()
            fs.addWidget(QtWidgets.QLabel("Text")); fs.addWidget(self.spin_font)
            fs.addWidget(QtWidgets.QLabel("Title")); fs.addWidget(self.spin_title_font)
            fs.addStretch(1)
            form.addRow("Font sizes:", fs)

            form.addRow("Face color:", self.btn_face)
            form.addRow("Edge color:", self.btn_edge)
            form.addRow("Text color:", self.btn_text)
            form.addRow("Frame alpha:", self.spin_alpha)

            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(form)
            lay.addWidget(QtWidgets.QLabel("Series labels"))
            lay.addWidget(self.labels_table, 1)
            lay.addWidget(btns)

            # init from parent widgets (these widgets are kept but not shown on main GUI)
            self.chk_show.setChecked(self.parent.chk_legend.isChecked())
            self.ed_title.setText(self.parent.ed_legend_title.text())
            self.combo_loc.setCurrentText(self.parent.combo_legend_loc.currentText())
            self.spin_font.setValue(self.parent.spin_legend_font.value())
            self.spin_title_font.setValue(self.parent.spin_legend_title_font.value())
            self.spin_alpha.setValue(self.parent.spin_legend_alpha.value())

            # colors
            self._set_color_btn(self.btn_face, self.parent._btn_hex(self.parent.btn_legend_face, "#FFFFFF"))
            self._set_color_btn(self.btn_edge, self.parent._btn_hex(self.parent.btn_legend_edge, "#000000"))
            self._set_color_btn(self.btn_text, self.parent._btn_hex(self.parent.btn_legend_text, "#000000"))

            self.btn_face.clicked.connect(lambda: self._pick_color(self.btn_face, "Pick legend face color"))
            self.btn_edge.clicked.connect(lambda: self._pick_color(self.btn_edge, "Pick legend edge color"))
            self.btn_text.clicked.connect(lambda: self._pick_color(self.btn_text, "Pick legend text color"))

            self._load_labels()

        def _set_color_btn(self, btn: QtWidgets.QPushButton, color: str):
            btn.setProperty("hexcolor", color)
            btn.setStyleSheet(f"background-color: {color};")

        def _pick_color(self, btn: QtWidgets.QPushButton, title: str):
            current = str(btn.property("hexcolor") or "#000000")
            chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, title)
            if chosen.isValid():
                self._set_color_btn(btn, chosen.name())

        def _load_labels(self):
            tbl = self.parent.styles_table
            self.labels_table.setRowCount(tbl.rowCount())
            for r in range(tbl.rowCount()):
                series_name = tbl.item(r, 0).text() if tbl.item(r, 0) else ""
                leg_name = tbl.item(r, 1).text() if tbl.item(r, 1) else series_name

                it0 = QtWidgets.QTableWidgetItem(series_name)
                it0.setFlags(it0.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)  # type: ignore
                self.labels_table.setItem(r, 0, it0)
                self.labels_table.setItem(r, 1, QtWidgets.QTableWidgetItem(leg_name))

        def apply_to_parent(self):
            p = self.parent
            p.chk_legend.setChecked(self.chk_show.isChecked())
            p.ed_legend_title.setText(self.ed_title.text())
            p.combo_legend_loc.setCurrentText(self.combo_loc.currentText())
            p.spin_legend_font.setValue(int(self.spin_font.value()))
            p.spin_legend_title_font.setValue(int(self.spin_title_font.value()))
            p.spin_legend_alpha.setValue(float(self.spin_alpha.value()))

            # write colors back to hidden buttons
            p._set_color_button(p.btn_legend_face, str(self.btn_face.property("hexcolor") or "#FFFFFF"))
            p._set_color_button(p.btn_legend_edge, str(self.btn_edge.property("hexcolor") or "#000000"))
            p._set_color_button(p.btn_legend_text, str(self.btn_text.property("hexcolor") or "#000000"))

            # write legend labels back to styles table
            p.styles_table.blockSignals(True)
            for r in range(p.styles_table.rowCount()):
                new_lab = self.labels_table.item(r, 1).text() if self.labels_table.item(r, 1) else ""
                if p.styles_table.item(r, 1):
                    p.styles_table.item(r, 1).setText(new_lab)
            p.styles_table.blockSignals(False)

            p.schedule_preview()

    class AxisEditDialog(QtWidgets.QDialog):
        """Edits axis label + axis/text colors. Intended to be opened by double-clicking an axis."""

        def __init__(self, parent: "MainWindow", axis: mpl.axes.Axes):
            super().__init__(parent)
            self.setWindowTitle("Edit axis")
            self.parent = parent
            self.axis = axis
            self.setModal(True)

            # infer which timestamp set this axis represents (overplot mode labels use '(S#)')
            self.set_id: Optional[int] = None
            # Prefer explicit axis tagging (we attach _ts_set_id when creating multi-x axes)
            sid = getattr(axis, "_ts_set_id", None)
            if isinstance(sid, (int, np.integer)):
                self.set_id = int(sid)

            # fields
            self.ed_xlabel = QtWidgets.QLineEdit()
            self.ed_ylabel = QtWidgets.QLineEdit()

            # colors
            self.btn_xaxis = QtWidgets.QPushButton("Pick…")   # per-set x-axis color
            self.btn_ticks = QtWidgets.QPushButton("Pick…")   # global tick color
            self.btn_labels = QtWidgets.QPushButton("Pick…")  # global label color
            self.btn_spines = QtWidgets.QPushButton("Pick…")  # global spine color
            self.btn_text = QtWidgets.QPushButton("Pick…")    # global text color
            self.btn_title = QtWidgets.QPushButton("Pick…")   # global title color

            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)

            form = QtWidgets.QFormLayout()
            form.addRow("X label:", self.ed_xlabel)
            form.addRow("Y label:", self.ed_ylabel)
            if self.set_id is not None and self.set_id in parent.xaxis_color_buttons:
                form.addRow(f"X axis color (S{self.set_id}):", self.btn_xaxis)
            else:
                self.btn_xaxis.setEnabled(False)

            form.addRow("Tick text color:", self.btn_ticks)
            form.addRow("Label text color:", self.btn_labels)
            form.addRow("Axis spine color:", self.btn_spines)
            form.addRow("All text color:", self.btn_text)
            form.addRow("Title text color:", self.btn_title)

            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(form)
            lay.addWidget(btns)

            # init from parent widgets
            self.ed_xlabel.setText(parent.ed_xlabel.text())
            self.ed_ylabel.setText(parent.ed_ylabel.text())

            if self.set_id is not None and self.set_id in parent.xaxis_color_buttons:
                self._set_color_btn(self.btn_xaxis, parent._btn_hex(parent.xaxis_color_buttons[self.set_id], "#000000"))

            self._set_color_btn(self.btn_ticks, parent._btn_hex(parent.btn_tick_color, "#000000"))
            self._set_color_btn(self.btn_labels, parent._btn_hex(parent.btn_label_color, "#000000"))
            self._set_color_btn(self.btn_spines, parent._btn_hex(parent.btn_spine_color, "#000000"))
            self._set_color_btn(self.btn_text, parent._btn_hex(parent.btn_text_color, "#000000"))
            self._set_color_btn(self.btn_title, parent._btn_hex(parent.btn_title_color, "#000000"))

            self.btn_xaxis.clicked.connect(lambda: self._pick_color(self.btn_xaxis, "Pick X-axis color"))
            self.btn_ticks.clicked.connect(lambda: self._pick_color(self.btn_ticks, "Pick tick color"))
            self.btn_labels.clicked.connect(lambda: self._pick_color(self.btn_labels, "Pick label color"))
            self.btn_spines.clicked.connect(lambda: self._pick_color(self.btn_spines, "Pick spine color"))
            self.btn_text.clicked.connect(lambda: self._pick_color(self.btn_text, "Pick text color"))
            self.btn_title.clicked.connect(lambda: self._pick_color(self.btn_title, "Pick title color"))

        def _set_color_btn(self, btn: QtWidgets.QPushButton, color: str):
            btn.setProperty("hexcolor", color)
            btn.setStyleSheet(f"background-color: {color};")

        def _pick_color(self, btn: QtWidgets.QPushButton, title: str):
            current = str(btn.property("hexcolor") or "#000000")
            chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, title)
            if chosen.isValid():
                self._set_color_btn(btn, chosen.name())

        def apply_to_parent(self):
            p = self.parent
            p.ed_xlabel.setText(self.ed_xlabel.text())
            p.ed_ylabel.setText(self.ed_ylabel.text())

            if self.set_id is not None and self.set_id in p.xaxis_color_buttons and self.btn_xaxis.isEnabled():
                p._set_color_button(p.xaxis_color_buttons[self.set_id], str(self.btn_xaxis.property("hexcolor") or "#000000"))

            p._set_color_button(p.btn_tick_color, str(self.btn_ticks.property("hexcolor") or "#000000"))
            p._set_color_button(p.btn_label_color, str(self.btn_labels.property("hexcolor") or "#000000"))
            p._set_color_button(p.btn_spine_color, str(self.btn_spines.property("hexcolor") or "#000000"))
            p._set_color_button(p.btn_text_color, str(self.btn_text.property("hexcolor") or "#000000"))
            p._set_color_button(p.btn_title_color, str(self.btn_title.property("hexcolor") or "#000000"))

            p.schedule_preview()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Reveal time-series GIF builder")

            self.df: Optional[pd.DataFrame] = None
            self.in_path: Optional[Path] = None

            # --- Keep these as state even though the corresponding boxes won't be shown ---
            # Colors
            self.btn_text_color = QtWidgets.QPushButton("Pick…")
            self.btn_title_color = QtWidgets.QPushButton("Pick…")
            self.btn_label_color = QtWidgets.QPushButton("Pick…")
            self.btn_tick_color = QtWidgets.QPushButton("Pick…")
            self.btn_spine_color = QtWidgets.QPushButton("Pick…")

            # Legend
            self.chk_legend = QtWidgets.QCheckBox("Show legend"); self.chk_legend.setChecked(True)
            self.ed_legend_title = QtWidgets.QLineEdit("")
            self.combo_legend_loc = QtWidgets.QComboBox()
            self.combo_legend_loc.addItems([
                "best","upper right","upper left","lower left","lower right","center right","center left",
                "upper center","lower center","center"
            ])
            self.spin_legend_font = QtWidgets.QSpinBox(); self.spin_legend_font.setRange(6, 72); self.spin_legend_font.setValue(10)
            self.spin_legend_title_font = QtWidgets.QSpinBox(); self.spin_legend_title_font.setRange(6, 72); self.spin_legend_title_font.setValue(10)
            self.btn_legend_face = QtWidgets.QPushButton("Pick…")
            self.btn_legend_edge = QtWidgets.QPushButton("Pick…")
            self.btn_legend_text = QtWidgets.QPushButton("Pick…")
            self.spin_legend_alpha = QtWidgets.QDoubleSpinBox(); self.spin_legend_alpha.setRange(0.0, 1.0); self.spin_legend_alpha.setSingleStep(0.05); self.spin_legend_alpha.setValue(0.9)

            # ---- Top / file ----
            self.btn_pick = QtWidgets.QPushButton("Select data file…")
            self.lbl_file = QtWidgets.QLabel("No file selected")
            self.lbl_file.setWordWrap(True)

            # ---- timestamp sets ----
            self.spin_timestamps = QtWidgets.QSpinBox()
            self.spin_timestamps.setMinimum(1)
            self.spin_timestamps.setMaximum(20)
            self.spin_timestamps.setValue(1)

            self.rows_widget = QtWidgets.QWidget()
            self.rows_layout = QtWidgets.QVBoxLayout(self.rows_widget)
            self.rows_layout.setContentsMargins(0, 0, 0, 0)
            self.rows_layout.setSpacing(8)
            self.rows: List[TimestampRow] = []

            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setWidget(self.rows_widget)
            self.scroll.setMinimumHeight(360)

            # ---- plot layout ----
            self.rb_overplot = QtWidgets.QRadioButton("Overplot (all series in one plot)")
            self.rb_stacked = QtWidgets.QRadioButton("Stacked (one subplot per series)")
            self.rb_overplot.setChecked(True)

            # ---- output / gif options ----
            self.out_path = QtWidgets.QLineEdit(str(Path.cwd() / "reveal.gif"))
            self.btn_out = QtWidgets.QPushButton("Browse…")

            self.spin_fps = QtWidgets.QSpinBox(); self.spin_fps.setRange(1, 120); self.spin_fps.setValue(30)
            self.spin_duration = QtWidgets.QDoubleSpinBox(); self.spin_duration.setRange(0.1, 600.0); self.spin_duration.setValue(12.0); self.spin_duration.setDecimals(2)
            self.combo_time_mode = QtWidgets.QComboBox(); self.combo_time_mode.addItems(["auto", "datetime", "numeric"])

            self.combo_xaxis_mode = QtWidgets.QComboBox()
            self.combo_xaxis_mode.addItems(["single_bottom","single_top","multi_top","multi_bottom","first_bottom_others_top"])

            self.combo_bg = QtWidgets.QComboBox(); self.combo_bg.addItems(["white", "transparent"])
            self.spin_grid_alpha = QtWidgets.QDoubleSpinBox(); self.spin_grid_alpha.setRange(0.0, 1.0); self.spin_grid_alpha.setSingleStep(0.05); self.spin_grid_alpha.setValue(0.25)
            self.ed_title = QtWidgets.QLineEdit("")
            self.ed_xlabel = QtWidgets.QLineEdit("Time")
            self.ed_ylabel = QtWidgets.QLineEdit("Value")
            self.spin_width = QtWidgets.QSpinBox(); self.spin_width.setRange(200, 8000); self.spin_width.setValue(1280)
            self.spin_height = QtWidgets.QSpinBox(); self.spin_height.setRange(200, 8000); self.spin_height.setValue(720)

            self.chk_loop = QtWidgets.QCheckBox("Loop GIF forever"); self.chk_loop.setChecked(True)

            # ---- Design ----
            self.font_combo = QtWidgets.QFontComboBox()
            self.spin_font = QtWidgets.QSpinBox(); self.spin_font.setRange(6, 48); self.spin_font.setValue(10)
            self.spin_title_font = QtWidgets.QSpinBox(); self.spin_title_font.setRange(6, 72); self.spin_title_font.setValue(14)
            self.spin_label_font = QtWidgets.QSpinBox(); self.spin_label_font.setRange(6, 72); self.spin_label_font.setValue(11)
            self.spin_tick_font = QtWidgets.QSpinBox(); self.spin_tick_font.setRange(6, 48); self.spin_tick_font.setValue(10)

            self.combo_default_type = QtWidgets.QComboBox(); self.combo_default_type.addItems(PLOT_TYPES)
            self.combo_default_marker = QtWidgets.QComboBox(); self.combo_default_marker.addItems(MARKERS)
            self.spin_default_marker = QtWidgets.QDoubleSpinBox(); self.spin_default_marker.setRange(1.0, 400.0); self.spin_default_marker.setValue(20.0)
            self.spin_default_lw = QtWidgets.QDoubleSpinBox(); self.spin_default_lw.setRange(0.1, 20.0); self.spin_default_lw.setValue(2.0)
            self.btn_apply_defaults = QtWidgets.QPushButton("Apply defaults to all")

            # Per-set x axis colors
            self.xaxis_colors_box = QtWidgets.QGroupBox("Per timestamp-set X-axis colors")
            self.xaxis_colors_layout = QtWidgets.QFormLayout(self.xaxis_colors_box)
            self.xaxis_color_buttons: Dict[int, QtWidgets.QPushButton] = {}

            # ---- Per-series style table ----
            self.styles_table = QtWidgets.QTableWidget(0, 8)
            self.styles_table.setHorizontalHeaderLabels([
                "Series", "Legend label", "Plot type", "Color", "Marker", "Marker size", "Line width", "Alpha"
            ])
            self.styles_table.horizontalHeader().setStretchLastSection(True)
            self.styles_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

            # ---- Preview ----
            self.preview_fig = Figure(figsize=(6, 3), dpi=100)
            self.preview_canvas = FigureCanvas(self.preview_fig)
            self.preview_canvas.setMinimumHeight(320)
            self._preview_timer = QtCore.QTimer()
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(250)

            # Track legend for hit testing
            self._preview_legend = None

            # ---- Build / progress / log ----
            self.btn_build = QtWidgets.QPushButton("Build GIF")
            self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
            self.progress_lbl = QtWidgets.QLabel("")
            self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)

            self._build_ui()
            self._wire_events()
            self._init_color_buttons()
            self._ensure_rows()
            self.schedule_preview()

            # double-click interactions on preview
            self.preview_canvas.mpl_connect("button_press_event", self._on_preview_click)

        # ---------- UI helpers ----------
        def _append_log(self, msg: str):
            self.log.appendPlainText(msg)

        def _error(self, title: str, message: str):
            QtWidgets.QMessageBox.critical(self, title, message)

        def _set_color_button(self, btn: QtWidgets.QPushButton, color: str):
            btn.setProperty("hexcolor", color)
            btn.setStyleSheet(f"background-color: {color};")

        def _btn_hex(self, btn: QtWidgets.QPushButton, default: str = "#000000") -> str:
            c = btn.property("hexcolor")
            return str(c) if c else default

        def _init_color_buttons(self):
            self._set_color_button(self.btn_text_color, "#000000")
            self._set_color_button(self.btn_title_color, "#000000")
            self._set_color_button(self.btn_label_color, "#000000")
            self._set_color_button(self.btn_tick_color, "#000000")
            self._set_color_button(self.btn_spine_color, "#000000")
            self._set_color_button(self.btn_legend_face, "#FFFFFF")
            self._set_color_button(self.btn_legend_edge, "#000000")
            self._set_color_button(self.btn_legend_text, "#000000")

        # ---------- build UI ----------
        def _build_ui(self):
            # Whole-window scroll area (kept from your scroll-fix)
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            central = QtWidgets.QWidget()
            scroll_area.setWidget(central)
            self.setCentralWidget(scroll_area)

            layout = QtWidgets.QVBoxLayout(central)

            file_row = QtWidgets.QHBoxLayout()
            file_row.addWidget(self.btn_pick)
            file_row.addWidget(self.lbl_file, 1)
            layout.addLayout(file_row)

            ts_row = QtWidgets.QHBoxLayout()
            ts_row.addWidget(QtWidgets.QLabel("How many timestamp sets are available?"))
            ts_row.addWidget(self.spin_timestamps)
            ts_row.addStretch(1)
            layout.addLayout(ts_row)

            layout.addWidget(self.scroll, 2)

            plot_box = QtWidgets.QGroupBox("Plot layout")
            plot_l = QtWidgets.QVBoxLayout(plot_box)
            plot_l.addWidget(self.rb_overplot)
            plot_l.addWidget(self.rb_stacked)
            layout.addWidget(plot_box)

            opt = QtWidgets.QGroupBox("GIF options")
            form = QtWidgets.QFormLayout(opt)
            out_row = QtWidgets.QHBoxLayout()
            out_row.addWidget(self.out_path, 1)
            out_row.addWidget(self.btn_out)
            form.addRow("Output GIF:", out_row)
            form.addRow("FPS:", self.spin_fps)
            form.addRow("Duration (seconds):", self.spin_duration)
            form.addRow("Time parsing:", self.combo_time_mode)
            form.addRow("X-axis handling:", self.combo_xaxis_mode)
            form.addRow("", self.chk_loop)
            form.addRow("Background:", self.combo_bg)
            form.addRow("Grid alpha:", self.spin_grid_alpha)
            form.addRow("Title:", self.ed_title)
            form.addRow("X label:", self.ed_xlabel)
            form.addRow("Y label (overplot mode):", self.ed_ylabel)
            size_row = QtWidgets.QHBoxLayout()
            size_row.addWidget(QtWidgets.QLabel("W")); size_row.addWidget(self.spin_width)
            size_row.addWidget(QtWidgets.QLabel("H")); size_row.addWidget(self.spin_height)
            size_row.addStretch(1)
            form.addRow("Size (px):", size_row)
            layout.addWidget(opt)

            design = QtWidgets.QGroupBox("Design")
            dform = QtWidgets.QFormLayout(design)
            dform.addRow("Font:", self.font_combo)
            font_row = QtWidgets.QHBoxLayout()
            font_row.addWidget(QtWidgets.QLabel("Base")); font_row.addWidget(self.spin_font)
            font_row.addWidget(QtWidgets.QLabel("Title")); font_row.addWidget(self.spin_title_font)
            font_row.addWidget(QtWidgets.QLabel("Labels")); font_row.addWidget(self.spin_label_font)
            font_row.addWidget(QtWidgets.QLabel("Ticks")); font_row.addWidget(self.spin_tick_font)
            font_row.addStretch(1)
            dform.addRow("Font sizes:", font_row)

            defaults_row = QtWidgets.QHBoxLayout()
            defaults_row.addWidget(QtWidgets.QLabel("Default type")); defaults_row.addWidget(self.combo_default_type)
            defaults_row.addWidget(QtWidgets.QLabel("Marker")); defaults_row.addWidget(self.combo_default_marker)
            defaults_row.addWidget(QtWidgets.QLabel("Marker size")); defaults_row.addWidget(self.spin_default_marker)
            defaults_row.addWidget(QtWidgets.QLabel("Line width")); defaults_row.addWidget(self.spin_default_lw)
            defaults_row.addStretch(1)
            dform.addRow("Defaults:", defaults_row)
            dform.addRow("", self.btn_apply_defaults)
            layout.addWidget(design)

            # NOTE: legend box + axis/text color box are intentionally NOT placed on the main GUI anymore.
            # They are now edited via double-click popups on the preview.

            # keep per-set x-axis colors box hidden but available for the axis popup
            layout.addWidget(self.xaxis_colors_box)
            self.xaxis_colors_box.setVisible(False)

            prevbox = QtWidgets.QGroupBox("Preview (double-click legend or axis to edit)")
            pl = QtWidgets.QVBoxLayout(prevbox)
            pl.addWidget(self.preview_canvas)
            layout.addWidget(prevbox, 2)

            st = QtWidgets.QGroupBox("Per-series styles")
            st_l = QtWidgets.QVBoxLayout(st)
            st_l.addWidget(self.styles_table)
            layout.addWidget(st, 2)

            btn_row = QtWidgets.QHBoxLayout()
            btn_row.addWidget(self.btn_build)
            btn_row.addWidget(self.progress, 1)
            btn_row.addWidget(self.progress_lbl)
            layout.addLayout(btn_row)

            log_box = QtWidgets.QGroupBox("Log")
            log_l = QtWidgets.QVBoxLayout(log_box)
            log_l.addWidget(self.log)
            layout.addWidget(log_box, 1)

        # ---------- popups via double click ----------
        def _legend_hit(self, event) -> bool:
            if self._preview_legend is None:
                return False
            try:
                renderer = self.preview_canvas.get_renderer()
            except Exception:
                renderer = None
            try:
                bbox = self._preview_legend.get_window_extent(renderer=renderer)
                return bbox.contains(event.x, event.y)
            except Exception:
                return False

        def _on_preview_click(self, event):
            # matplotlib event; event.dblclick exists for button_press_event
            if not getattr(event, "dblclick", False):
                return

            # prefer legend
            if self._legend_hit(event):
                dlg = LegendEditDialog(self)
                if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    dlg.apply_to_parent()
                return

            # axis double-click
            if getattr(event, "inaxes", None) is not None:
                dlg = AxisEditDialog(self, event.inaxes)
                if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    dlg.apply_to_parent()
                return

        # ---------- wiring ----------
        def _wire_events(self):
            self.btn_pick.clicked.connect(self.on_pick)
            self.btn_out.clicked.connect(self.on_pick_out)
            self.btn_build.clicked.connect(self.on_build)

            self.spin_timestamps.valueChanged.connect(lambda _: self._ensure_rows())
            self.btn_apply_defaults.clicked.connect(self.apply_defaults_to_all)

            # preview timer
            self._preview_timer.timeout.connect(self.render_preview)

            # schedule preview on changes (legend/colors are edited via popups but still stored on hidden widgets)
            for w in [
                self.rb_overplot, self.rb_stacked,
                self.spin_fps, self.spin_duration,
                self.combo_time_mode, self.combo_xaxis_mode,
                self.combo_bg, self.spin_grid_alpha,
                self.ed_title, self.ed_xlabel, self.ed_ylabel,
                self.spin_width, self.spin_height,
                self.font_combo,
                self.spin_font, self.spin_title_font, self.spin_label_font, self.spin_tick_font,
                self.combo_default_type, self.combo_default_marker,
                self.spin_default_marker, self.spin_default_lw,
                self.chk_loop,
            ]:
                if hasattr(w, "toggled"):
                    w.toggled.connect(lambda *_: self.schedule_preview())
                if hasattr(w, "valueChanged"):
                    w.valueChanged.connect(lambda *_: self.schedule_preview())
                if hasattr(w, "currentTextChanged"):
                    w.currentTextChanged.connect(lambda *_: self.schedule_preview())
                if hasattr(w, "textChanged"):
                    w.textChanged.connect(lambda *_: self.schedule_preview())
                if hasattr(w, "currentFontChanged"):
                    w.currentFontChanged.connect(lambda *_: self.schedule_preview())

            self.styles_table.itemChanged.connect(lambda *_: self.schedule_preview())

        # ---------- timestamp rows ----------
        def _ensure_rows(self):
            needed = int(self.spin_timestamps.value())

            while self.rows_layout.count():
                item = self.rows_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

            self.rows = self.rows[:needed]
            while len(self.rows) < needed:
                r = TimestampRow(f"Timestamp set {len(self.rows) + 1}")
                r.changed.connect(self.refresh_styles_table)
                r.changed.connect(self.schedule_preview)
                self.rows.append(r)

            for r in self.rows:
                self.rows_layout.addWidget(r)
            self.rows_layout.addStretch(1)

            # rebuild x-axis color buttons (hidden, but used by axis popup and rendering)
            while self.xaxis_colors_layout.count():
                it = self.xaxis_colors_layout.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None)
            self.xaxis_color_buttons = {}
            for sid in range(1, needed + 1):
                btn = QtWidgets.QPushButton("Pick…")
                self._set_color_button(btn, "#000000")
                self.xaxis_color_buttons[sid] = btn
                self.xaxis_colors_layout.addRow(f"X axis color for S{sid}:", btn)

            if self.df is not None:
                cols = [str(c) for c in self.df.columns]
                for r in self.rows:
                    r.set_columns(cols)
                self._autopick_defaults()
                self.refresh_styles_table()

        def _autopick_defaults(self):
            if self.df is None:
                return
            cols = [str(c) for c in self.df.columns]
            if not cols:
                return

            def default_x() -> str:
                for c in cols:
                    name = c.lower()
                    if "time" in name or "date" in name or "timestamp" in name:
                        return c
                return cols[0]

            def numeric_candidates(exclude: str) -> List[str]:
                out = []
                for c in cols:
                    if c == exclude:
                        continue
                    s = coerce_numeric_series(self.df[c])
                    if s.notna().sum() >= max(3, int(0.3 * len(self.df))):
                        out.append(c)
                return out

            for r in self.rows:
                x = default_x()
                r.x_combo.setCurrentText(x)
                cands = numeric_candidates(x)
                for i in range(min(2, len(cands))):
                    items = r.y_list.findItems(cands[i], _qt_match_exact_flag())
                    for it in items:
                        it.setSelected(True)

        # ---------- series/style table ----------
        def _current_series_labels(self) -> List[str]:
            labels: List[str] = []
            if self.df is None:
                return labels
            for set_idx, r in enumerate(self.rows, start=1):
                _, y_cols = r.get_selection()
                for y_col in y_cols:
                    labels.append(f"{y_col}" if len(self.rows) == 1 else f"S{set_idx}: {y_col}")
            return labels

        def refresh_styles_table(self):
            labels = self._current_series_labels()
            if not labels:
                self.styles_table.setRowCount(0)
                return

            old = self._read_style_table_by_label()
            self.styles_table.blockSignals(True)
            self.styles_table.setRowCount(len(labels))

            for row, label in enumerate(labels):
                item = QtWidgets.QTableWidgetItem(label)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)  # type: ignore
                self.styles_table.setItem(row, 0, item)

                leg_item = QtWidgets.QTableWidgetItem(label)
                self.styles_table.setItem(row, 1, leg_item)

                cb_type = QtWidgets.QComboBox(); cb_type.addItems(PLOT_TYPES)
                self.styles_table.setCellWidget(row, 2, cb_type)

                btn_color = QtWidgets.QPushButton("Pick…")
                self.styles_table.setCellWidget(row, 3, btn_color)

                cb_marker = QtWidgets.QComboBox(); cb_marker.addItems(MARKERS)
                self.styles_table.setCellWidget(row, 4, cb_marker)

                sp_ms = QtWidgets.QDoubleSpinBox(); sp_ms.setRange(1.0, 400.0)
                self.styles_table.setCellWidget(row, 5, sp_ms)

                sp_lw = QtWidgets.QDoubleSpinBox(); sp_lw.setRange(0.1, 20.0)
                self.styles_table.setCellWidget(row, 6, sp_lw)

                sp_a = QtWidgets.QDoubleSpinBox(); sp_a.setRange(0.0, 1.0); sp_a.setSingleStep(0.05)
                self.styles_table.setCellWidget(row, 7, sp_a)

                cb_type.setCurrentText(str(self.combo_default_type.currentText()))
                cb_marker.setCurrentText(str(self.combo_default_marker.currentText()))
                sp_ms.setValue(float(self.spin_default_marker.value()))
                sp_lw.setValue(float(self.spin_default_lw.value()))
                sp_a.setValue(1.0)
                color = _default_color_cycle(len(labels))[row]
                self._set_color_button(btn_color, color)

                if label in old:
                    st = old[label]
                    leg_item.setText(str(st.get("legend_label", label)))
                    cb_type.setCurrentText(str(st.get("plot_type", "markers")))
                    cb_marker.setCurrentText(str(st.get("marker", "o")))
                    sp_ms.setValue(float(st.get("marker_size", 20.0)))
                    sp_lw.setValue(float(st.get("line_width", 2.0)))
                    sp_a.setValue(float(st.get("alpha", 1.0)))
                    self._set_color_button(btn_color, str(st.get("color", color)))

                btn_color.clicked.connect(lambda _=False, r=row: self.pick_color_for_row(r))
                cb_type.currentTextChanged.connect(lambda *_: self.schedule_preview())
                cb_marker.currentTextChanged.connect(lambda *_: self.schedule_preview())
                sp_ms.valueChanged.connect(lambda *_: self.schedule_preview())
                sp_lw.valueChanged.connect(lambda *_: self.schedule_preview())
                sp_a.valueChanged.connect(lambda *_: self.schedule_preview())

                def _sync_controls(_=None, r=row):
                    t = self._get_row_plot_type(r)
                    is_bar = t == "bar"
                    is_line = t in ("line", "line+markers")
                    is_mark = t in ("markers", "line+markers")
                    self._row_marker_combo(r).setEnabled(is_mark)
                    self._row_marker_size_spin(r).setEnabled(is_mark)
                    self._row_linewidth_spin(r).setEnabled(is_line)
                    if is_bar:
                        self._row_marker_combo(r).setEnabled(False)
                        self._row_marker_size_spin(r).setEnabled(False)
                        self._row_linewidth_spin(r).setEnabled(False)

                cb_type.currentTextChanged.connect(_sync_controls)
                _sync_controls()

            self.styles_table.blockSignals(False)

        def _row_widget(self, row: int, col: int):
            return self.styles_table.cellWidget(row, col)

        def _get_row_legend_label(self, row: int) -> str:
            it = self.styles_table.item(row, 1)
            return it.text() if it else ""

        def _get_row_plot_type(self, row: int) -> str:
            cb = self._row_widget(row, 2)
            return str(cb.currentText()) if cb else "markers"

        def _get_row_color(self, row: int) -> str:
            btn = self._row_widget(row, 3)
            c = btn.property("hexcolor") if btn else None
            return str(c) if c else "#1f77b4"

        def _row_marker_combo(self, row: int):
            return self._row_widget(row, 4)

        def _row_marker_size_spin(self, row: int):
            return self._row_widget(row, 5)

        def _row_linewidth_spin(self, row: int):
            return self._row_widget(row, 6)

        def _row_alpha_spin(self, row: int):
            return self._row_widget(row, 7)

        def _read_style_table_by_label(self) -> Dict[str, Dict[str, object]]:
            out: Dict[str, Dict[str, object]] = {}
            for row in range(self.styles_table.rowCount()):
                label_item = self.styles_table.item(row, 0)
                if not label_item:
                    continue
                label = label_item.text()
                out[label] = {
                    "legend_label": self._get_row_legend_label(row),
                    "plot_type": self._get_row_plot_type(row),
                    "color": self._get_row_color(row),
                    "marker": str(self._row_marker_combo(row).currentText()) if self._row_marker_combo(row) else "o",
                    "marker_size": float(self._row_marker_size_spin(row).value()) if self._row_marker_size_spin(row) else 20.0,
                    "line_width": float(self._row_linewidth_spin(row).value()) if self._row_linewidth_spin(row) else 2.0,
                    "alpha": float(self._row_alpha_spin(row).value()) if self._row_alpha_spin(row) else 1.0,
                }
            return out

        def pick_color_for_row(self, row: int):
            current = self._get_row_color(row)
            chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, "Pick series color")
            if not chosen.isValid():
                return
            btn = self._row_widget(row, 3)
            if btn is not None:
                self._set_color_button(btn, chosen.name())
            self.schedule_preview()

        def apply_defaults_to_all(self):
            for row in range(self.styles_table.rowCount()):
                cb_type = self._row_widget(row, 2)
                cb_marker = self._row_marker_combo(row)
                sp_ms = self._row_marker_size_spin(row)
                sp_lw = self._row_linewidth_spin(row)
                if cb_type:
                    cb_type.setCurrentText(str(self.combo_default_type.currentText()))
                if cb_marker:
                    cb_marker.setCurrentText(str(self.combo_default_marker.currentText()))
                if sp_ms:
                    sp_ms.setValue(float(self.spin_default_marker.value()))
                if sp_lw:
                    sp_lw.setValue(float(self.spin_default_lw.value()))
            self.schedule_preview()

        # ---------- build specs ----------
        def _build_style_spec(self) -> StyleSpec:
            bg = "white" if str(self.combo_bg.currentText()) == "white" else "none"
            return StyleSpec(
                font_family=str(self.font_combo.currentFont().family()),
                base_font_size=int(self.spin_font.value()),
                title_size=int(self.spin_title_font.value()),
                label_size=int(self.spin_label_font.value()),
                tick_size=int(self.spin_tick_font.value()),
                background=bg,
                grid_alpha=float(self.spin_grid_alpha.value()),
                text_color=self._btn_hex(self.btn_text_color),
                title_color=self._btn_hex(self.btn_title_color),
                label_color=self._btn_hex(self.btn_label_color),
                tick_color=self._btn_hex(self.btn_tick_color),
                spine_color=self._btn_hex(self.btn_spine_color),
            )

        def _build_legend_spec(self) -> LegendSpec:
            return LegendSpec(
                show=bool(self.chk_legend.isChecked()),
                loc=str(self.combo_legend_loc.currentText()),
                title=str(self.ed_legend_title.text()),
                fontsize=int(self.spin_legend_font.value()),
                title_fontsize=int(self.spin_legend_title_font.value()),
                frame=True,
                facecolor=self._btn_hex(self.btn_legend_face, "#FFFFFF"),
                edgecolor=self._btn_hex(self.btn_legend_edge, "#000000"),
                framealpha=float(self.spin_legend_alpha.value()),
                textcolor=self._btn_hex(self.btn_legend_text, "#000000"),
            )

        def _build_xaxis_colors(self) -> Dict[int, str]:
            return {sid: self._btn_hex(btn) for sid, btn in self.xaxis_color_buttons.items()}

        # ---------- preview ----------
        def schedule_preview(self):
            if self._preview_timer.isActive():
                self._preview_timer.stop()
            self._preview_timer.start()

        def render_preview(self):
            if self.df is None:
                self.preview_fig.clear()
                self.preview_canvas.draw()
                return
            try:
                series = self._collect_series_for_plot()
                style = self._build_style_spec()
                leg = self._build_legend_spec()
                xcols = self._build_xaxis_colors()
                mode = "overplot" if self.rb_overplot.isChecked() else "stacked"

                rc = {
                    "font.family": style.font_family,
                    "font.size": style.base_font_size,
                    "axes.titlesize": style.title_size,
                    "axes.labelsize": style.label_size,
                    "xtick.labelsize": style.tick_size,
                    "ytick.labelsize": style.tick_size,
                    "text.color": style.text_color,
                    "axes.labelcolor": style.label_color,
                    "xtick.color": style.tick_color,
                    "ytick.color": style.tick_color,
                    "axes.edgecolor": style.spine_color,
                }

                with mpl.rc_context(rc=rc):
                    draw_full_reveal_preview(
                        fig=self.preview_fig,
                        series=series,
                        mode=mode,
                        title=self.ed_title.text().strip(),
                        xlabel=self.ed_xlabel.text().strip() or "Time",
                        ylabel=self.ed_ylabel.text().strip() or "Value",
                        style=style,
                        legend=leg,
                        xaxis_mode=str(self.combo_xaxis_mode.currentText()),
                        xaxis_colors=xcols,
                    )

                # keep legend reference for hit-testing
                self._preview_legend = None
                if self.preview_fig.axes:
                    base_ax = self.preview_fig.axes[0]
                    self._preview_legend = base_ax.get_legend()
                self.preview_canvas.draw()
            except Exception as e:
                self._append_log(f"Preview skipped: {e}")

        # ---------- data collection ----------
        def _collect_series_for_plot(self) -> List[SeriesSpec]:
            if self.df is None:
                return []
            time_mode = str(self.combo_time_mode.currentText())
            self.refresh_styles_table()
            st = self._read_style_table_by_label()

            out: List[SeriesSpec] = []
            for set_idx, r in enumerate(self.rows, start=1):
                x_col, y_cols = r.get_selection()
                if not x_col or not y_cols:
                    continue
                x_num, x_dt = parse_x_series(self.df, x_col, time_mode)
                is_dt = x_dt is not None
                for y_col in y_cols:
                    y_num = coerce_numeric_series(self.df[y_col]).to_numpy()
                    label = f"{y_col}" if len(self.rows) == 1 else f"S{set_idx}: {y_col}"
                    cfg = st.get(label, {})
                    out.append(
                        SeriesSpec(
                            label=label,
                            x=x_num,
                            y=y_num,
                            set_id=set_idx,
                            is_datetime=is_dt,
                            legend_label=str(cfg.get("legend_label", label)),
                            plot_type=str(cfg.get("plot_type", "markers")),
                            color=str(cfg.get("color", None)) if cfg.get("color", None) is not None else None,
                            marker=str(cfg.get("marker", "o")),
                            marker_size=float(cfg.get("marker_size", float(self.spin_default_marker.value()))),
                            line_width=float(cfg.get("line_width", float(self.spin_default_lw.value()))),
                            alpha=float(cfg.get("alpha", 1.0)),
                        )
                    )
            return out

        # ---------- file pickers ----------
        def on_pick(self):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select a CSV / Excel / TXT file",
                "",
                "Data files (*.csv *.txt *.tsv *.xlsx *.xls);;All files (*.*)",
            )
            if not file_path:
                return
            self.in_path = Path(file_path)
            self.lbl_file.setText(str(self.in_path))
            try:
                self.df = load_table(self.in_path)
            except Exception as e:
                self.df = None
                self._error("Load failed", f"Could not load file.\n\n{e}")
                return

            self._append_log(f"Loaded: {self.in_path} ({len(self.df)} rows, {len(self.df.columns)} cols)")
            self._ensure_rows()
            self.schedule_preview()

        def on_pick_out(self):
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save GIF as",
                self.out_path.text().strip() or "reveal.gif",
                "GIF (*.gif)",
            )
            if out_path:
                if not out_path.lower().endswith(".gif"):
                    out_path += ".gif"
                self.out_path.setText(out_path)

        # ---------- build gif ----------
        def on_build(self):
            if self.df is None or self.in_path is None:
                self._error("Missing input", "Please select a data file first.")
                return

            series = self._collect_series_for_plot()
            if not series:
                self._error("Selection error", "Select at least one (x,y) series.")
                return

            out = Path(self.out_path.text().strip() or "reveal.gif").expanduser().resolve()
            dpi = 120
            figsize = (self.spin_width.value() / dpi, self.spin_height.value() / dpi)
            style = self._build_style_spec()
            leg = self._build_legend_spec()
            xcols = self._build_xaxis_colors()
            mode = "overplot" if self.rb_overplot.isChecked() else "stacked"

            self.progress.setValue(0)
            self.progress_lbl.setText("0%")
            self.btn_build.setEnabled(False)
            self._append_log(f"Building GIF: {out}")

            def on_progress(i: int, n: int):
                if n <= 0:
                    return
                pct = int(round(100 * (i / float(n))))
                pct = max(0, min(100, pct))
                self.progress.setValue(pct)
                self.progress_lbl.setText(f"{pct}%")
                QtWidgets.QApplication.processEvents()

            try:
                build_reveal_markers_gif_multi(
                    series=series,
                    out_path=out,
                    mode=mode,
                    fps=int(self.spin_fps.value()),
                    duration=float(self.spin_duration.value()),
                    dpi=dpi,
                    figsize=figsize,
                    title=self.ed_title.text().strip(),
                    xlabel=self.ed_xlabel.text().strip() or "Time",
                    ylabel=self.ed_ylabel.text().strip() or "Value",
                    style=style,
                    legend=leg,
                    xaxis_mode=str(self.combo_xaxis_mode.currentText()),
                    xaxis_colors=xcols,
                    progress_callback=on_progress,
                    loop_forever=bool(self.chk_loop.isChecked()),
                )
            except Exception as e:
                self._error("Build failed", f"Could not build GIF.\n\n{e}")
                self._append_log(f"ERROR: {e}")
                self.btn_build.setEnabled(True)
                return
            finally:
                self.btn_build.setEnabled(True)

            self.progress.setValue(100)
            self.progress_lbl.setText("100%")
            self._append_log(f"Saved GIF: {out}")
            QtWidgets.QMessageBox.information(self, "Done", f"Saved GIF:\n{out}")

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 980)
    w.show()
    return app.exec()


# ----------------------------
# CLI entry (kept)
# ----------------------------


def run_cli(args: argparse.Namespace) -> int:
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = load_table(in_path, sheet=args.sheet, delimiter=args.delimiter)
    if args.x not in df.columns:
        raise ValueError(f"x column '{args.x}' not found. Available: {list(df.columns)}")

    x_num, x_dt = parse_x_series(df, args.x, args.time_mode)
    is_dt = x_dt is not None

    y_cols = [c.strip() for c in args.y.split(",") if c.strip()]
    if not y_cols:
        raise ValueError("Provide at least one y column via --y (comma-separated allowed)")

    series: List[SeriesSpec] = []
    for y in y_cols:
        if y not in df.columns:
            raise ValueError(f"y column '{y}' not found. Available: {list(df.columns)}")
        y_num = coerce_numeric_series(df[y]).to_numpy()
        series.append(
            SeriesSpec(
                label=y,
                x=x_num,
                y=y_num,
                set_id=1,
                is_datetime=is_dt,
                legend_label=y,
                plot_type=args.plot_type,
                color=args.color,
                marker=args.marker,
                marker_size=args.marker_size,
                line_width=args.line_width,
                alpha=args.alpha,
            )
        )

    dpi = 120
    figsize = (args.width / dpi, args.height / dpi)
    bg = "white" if args.bg == "white" else "none"

    style = StyleSpec(
        font_family=args.font,
        base_font_size=args.font_size,
        title_size=args.title_size,
        label_size=args.label_size,
        tick_size=args.tick_size,
        background=bg,
        grid_alpha=args.grid_alpha,
    )

    legend = LegendSpec(show=not args.no_legend)

    build_reveal_markers_gif_multi(
        series=series,
        out_path=Path(args.out),
        mode=args.mode,
        fps=args.fps,
        duration=args.duration,
        dpi=dpi,
        figsize=figsize,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        style=style,
        legend=legend,
        xaxis_mode=args.xaxis_mode,
        xaxis_colors=None,
        progress_callback=None,
        loop_forever=args.loop,
    )
    print(f"Loaded: {in_path}")
    print(f"x='{args.x}' y={y_cols}")
    print(f"Saved GIF: {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--gui", action="store_true", help="Force launching the GUI")
    ap.add_argument("--input", default=None, help="Path to CSV / Excel / TXT")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--delimiter", default=None, help=r"CSV/TXT delimiter, e.g. ',' ';' '\t' '|'. Auto if omitted.")
    ap.add_argument("--x", default=None, help="X column name (time)")
    ap.add_argument("--y", default=None, help="Y column name(s). Comma-separated allowed.")
    ap.add_argument("--time-mode", choices=["auto", "datetime", "numeric"], default="auto")
    ap.add_argument("--mode", choices=["overplot", "stacked"], default="overplot")
    ap.add_argument("--out", default="reveal.gif", help="Output GIF path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Time")
    ap.add_argument("--ylabel", default="Value")
    ap.add_argument("--width", type=int, default=1280, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Output height in pixels")
    ap.add_argument("--bg", choices=["white", "transparent"], default="white")
    ap.add_argument("--xaxis-mode", choices=["single_bottom","single_top","multi_top","multi_bottom","first_bottom_others_top"], default="single_bottom")
    ap.add_argument("--loop", action="store_true", help="Loop GIF forever (default false in CLI)", default=False)
    ap.add_argument("--no-legend", action="store_true")

    # per-series defaults (CLI)
    ap.add_argument("--plot-type", choices=["markers", "line", "line+markers", "bar"], default="markers")
    ap.add_argument("--color", default=None)
    ap.add_argument("--marker", choices=["o", ".", "x", "+", "s", "^", "v", "D", "*", "P", "X"], default="o")
    ap.add_argument("--marker-size", type=float, default=20.0)
    ap.add_argument("--line-width", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)

    # global design (CLI)
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--font-size", type=int, default=10)
    ap.add_argument("--title-size", type=int, default=14)
    ap.add_argument("--label-size", type=int, default=11)
    ap.add_argument("--tick-size", type=int, default=10)
    ap.add_argument("--grid-alpha", type=float, default=0.25)

    args = ap.parse_args()

    has_cli = args.input is not None and args.x is not None and args.y is not None
    if args.gui or not has_cli:
        return run_gui()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())