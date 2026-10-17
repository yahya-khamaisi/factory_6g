"""Conceptual / motivation thesis figures (matplotlib, no simulation data).

Chapter 1: ``plot_urllc_tradeoff`` (service-class map).
Chapter 2 literature-review figure set (all matplotlib, one visual language):
    * ``plot_lr_taxonomy``           -- 2-D literature map; the empty joint cell is the gap.
    * ``plot_industry50``            -- Industry 4.0 to Industry 5.0 and the shared wireless need (S2.3).
    * ``plot_fading_scenarios``      -- Rayleigh / Rician / TR 38.901 fading intuition (P4).
    * ``plot_ce_estimator_tradeoff`` -- cost vs robustness of channel-estimation families (S2.4).
    * ``plot_drl_rm_loop``           -- DRL scheduling loop and where estimator error enters (S2.5).
    * ``plot_lr_eval_coupling``      -- PHY-only / MAC-only / coupled evaluation patterns (S2.7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from factory6g.visualization.thesis_plot_style import (
    FIG_CALLOUT_PT,
    INK,
    PALETTE,
    SEMANTIC,
    TABLE_FONT_PT,
    THESIS_DPI,
    apply_thesis_rcparams,
)

SUNWAY_NAVY = "#233369"
SUNWAY_GREY = "#5b6472"

# Semantic colours shared with the draw.io diagrams (thesis_plot_style.SEMANTIC):
# the same meaning in every figure of the thesis.
COL_CE = SEMANTIC["phy"][1]          # channel estimation / PHY  (blue)
COL_RM = SEMANTIC["mac"][1]          # resource management / MAC (orange)
COL_CLASSICAL = SEMANTIC["process"][1]  # classical / neutral    (grey)
COL_LEARNED = SEMANTIC["learned"][1]  # learned methods          (purple)
COL_HYBRID = SEMANTIC["output"][1]   # metrics, rewards, outputs (green)
COL_ENV = SEMANTIC["env"][1]         # environment and channel   (stone)
COL_GAP = SEMANTIC["coupling"][1]    # PHY-MAC coupling / gap    (vermillion)

# Stroke -> light fill for schematic boxes.
_FILL = {stroke: fill for fill, stroke in SEMANTIC.values()}


def _ink(color: str) -> str:
    """Schematic text/arrow colour: semantic colours stay, generic grey becomes ink."""
    return INK if color == SUNWAY_GREY else color

# Dense schematics need smaller type than the Chapter 1 plots: legibility here
# comes from boxes and spacing, not font size.
_S_TITLE = 12.5
_S_HEAD = 9.5
_S_BODY = 8.0
_S_NOTE = 7.3


def _save(fig: Any, path: Path, *, trim: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=THESIS_DPI, bbox_inches="tight")
    if trim:
        _trim_whitespace(path)
    print(f"Wrote {path}")


def _trim_whitespace(path: Path, pad: int = 24) -> None:
    """Crop uniform white margins from a schematic drawn in axes coordinates.

    ``bbox_inches="tight"`` keeps the full (invisible) axes box, so schematics
    without an in-figure title would otherwise carry an empty band at the top.
    """
    from PIL import Image

    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = np.where((255 - rgb).sum(axis=2) > 24)
    if ink[0].size == 0:
        return
    top, bottom = max(int(ink[0].min()) - pad, 0), min(int(ink[0].max()) + pad, rgb.shape[0])
    left, right = max(int(ink[1].min()) - pad, 0), min(int(ink[1].max()) + pad, rgb.shape[1])
    img.crop((left, top, right, bottom)).save(path, dpi=(THESIS_DPI, THESIS_DPI))


# --------------------------------------------------------------------------- #
# Shared drawing helpers (axes-fraction coordinates, transform=ax.transAxes). #
# --------------------------------------------------------------------------- #


def _box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float = 0.15,
    lw: float = 1.4,
    ls: str = "-",
    hatch: str | None = None,
    rounding: float = 0.015,
    zorder: float = 2.0,
    learned: bool = False,
) -> None:
    """Square box: white (or navy tint for the gap) with a thin ink border."""
    import matplotlib.patches as mpatches

    edgecolor = _ink(edgecolor)
    if w >= 0.6:
        # A box spanning most of the figure is a lane/group frame: role-coloured
        # outline, white inside, so the nodes it holds carry the fill.
        facecolor, alpha, lw = "white", 1.0, 1.0
    elif edgecolor in _FILL:
        facecolor, alpha = _FILL[edgecolor], 1.0
        lw = 2.0 if edgecolor == COL_LEARNED else min(max(lw, 1.0), 1.4)
    else:
        facecolor, alpha, lw = "white", 1.0, min(lw, 0.9)
    if learned:
        # Learned component: keep the layer fill, purple outline (as in draw.io).
        edgecolor, lw = COL_LEARNED, 2.0
    boxstyle = "square,pad=0.004"
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=boxstyle, linewidth=0, edgecolor="none",
            facecolor=facecolor, alpha=alpha, hatch=hatch,
            transform=ax.transAxes, zorder=zorder,
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=boxstyle, linewidth=lw, edgecolor=edgecolor,
            facecolor="none", linestyle=ls, transform=ax.transAxes, zorder=zorder + 0.2,
        )
    )


def _text(
    ax: Any,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: str = "#1a1a1a",
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
    rotation: float = 0.0,
    zorder: float = 5.0,
) -> None:
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=_ink(color),
        fontweight=weight, fontstyle=style, rotation=rotation, linespacing=1.35,
        zorder=zorder,
    )


def _arrow(
    ax: Any,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    *,
    color: str = SUNWAY_GREY,
    lw: float = 1.5,
    ls: str = "-",
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0.0",
    zorder: float = 3.0,
) -> None:
    ax.annotate(
        "", xy=xy_to, xytext=xy_from, xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle=style, color=_ink(color), lw=min(lw, 1.4) if _ink(color) == INK else lw, linestyle=ls,
                        connectionstyle=connectionstyle, shrinkA=1.0, shrinkB=1.0),
        zorder=zorder,
    )


def _title_block(ax: Any, title: str, subtitle: str, *, y: float = 0.99) -> None:
    _text(ax, 0.5, y, title, size=_S_TITLE, color=SUNWAY_NAVY, weight="bold", va="top")
    _text(ax, 0.5, y - 0.075, subtitle, size=_S_NOTE + 0.7, color=SUNWAY_GREY, va="top")


# --------------------------------------------------------------------------- #
# Chapter 1                                                                   #
# --------------------------------------------------------------------------- #


def plot_urllc_tradeoff(*, output_path: Path) -> None:
    """Reliability--latency design space with throughput as marker scale (motivation)."""
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    def _nines(reliability_pct: float) -> float:
        """Map a reliability percentage to its number of nines.

        A linear reliability axis collapses 99.9 %, 99.99 % and 99.999 % onto
        one another; plotting the number of nines separates the URLLC classes
        that the figure is about.
        """
        return float(-np.log10(1.0 - reliability_pct / 100.0))

    services = {
        "Factory URLLC\n(closed-loop control)": {
            "latency_ms": 1.0, "reliability_pct": 99.999, "throughput": 80,
            "xytext": (-4, 22), "ha": "center", "va": "bottom",
        },
        "URLLC (general)": {
            "latency_ms": 5.0, "reliability_pct": 99.99, "throughput": 120,
            "xytext": (52, 0), "ha": "left", "va": "center",
        },
        "eMBB": {
            "latency_ms": 40.0, "reliability_pct": 99.0, "throughput": 400,
            "xytext": (0, 26), "ha": "center", "va": "bottom",
        },
        "mMTC": {
            "latency_ms": 500.0, "reliability_pct": 99.0, "throughput": 60,
            "xytext": (0, 20), "ha": "center", "va": "bottom",
        },
    }

    y_lo, y_hi = 1.7, 5.4
    ax.set_ylim(y_lo, y_hi)

    def _yfrac(nines: float) -> float:
        return (nines - y_lo) / (y_hi - y_lo)

    ax.axvspan(0.3, 10.0, ymin=_yfrac(_nines(99.9)), ymax=1.0,
               color=SUNWAY_NAVY, alpha=0.08, zorder=0)
    ax.text(0.42, _nines(99.95), "Factory URLLC\ntarget band", fontsize=FIG_CALLOUT_PT - 2,
            color=SUNWAY_NAVY, ha="left", va="center", linespacing=1.2, zorder=2)

    for index, (label, spec) in enumerate(services.items()):
        color = PALETTE[index % len(PALETTE)]
        size = 40 + spec["throughput"] * 0.35
        marker_radius_pt = float(np.sqrt(size / np.pi))
        y_val = _nines(spec["reliability_pct"])
        ax.scatter(spec["latency_ms"], y_val, s=size, c=color,
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=label.replace("\n", " "))
        ax.annotate(
            label, xy=(spec["latency_ms"], y_val), xytext=spec["xytext"],
            textcoords="offset points", fontsize=FIG_CALLOUT_PT - 1,
            ha=spec.get("ha", "left"), va=spec.get("va", "center"), color=color,
            linespacing=1.2,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=2,
                            shrinkB=max(2.0, marker_radius_pt * 0.35), relpos=(0.5, 0.5)),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.4, 900)
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Reliability")
    tick_rel = [99.0, 99.9, 99.99, 99.999]
    ax.set_yticks([_nines(r) for r in tick_rel])
    ax.set_yticklabels([f"{r:g}%" for r in tick_rel])
    ax.grid(True, which="major")
    ax.annotate(
        "Marker area $\\propto$ throughput demand", xy=(0.97, 0.90),
        xycoords="axes fraction", fontsize=FIG_CALLOUT_PT - 2, color=SUNWAY_GREY,
        ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=SUNWAY_GREY, alpha=0.95),
        zorder=6,
    )

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Figure 2.1: literature map                                     #
# --------------------------------------------------------------------------- #


def plot_lr_taxonomy(*, output_path: Path) -> None:
    """2-D map of the reviewed literature (Ch2 SS2.1).

    Columns are the pipeline stage a study addresses; rows are the method class.
    The four single-stage cells are mature branches; the learned x joint cell is
    empty, and that emptiness is the research gap.
    """
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # No in-figure title: the LaTeX caption carries it (thesis figure convention).
    cols = {"ce": (0.115, 0.365), "rm": (0.405, 0.655), "joint": (0.695, 0.945)}
    rows = {"learned": (0.470, 0.720), "classical": (0.150, 0.400)}

    ax.add_patch(plt.Rectangle((0.675, 0.125), 0.29, 0.615, transform=ax.transAxes,
                               facecolor=COL_GAP, alpha=0.055, zorder=0))

    for key, label in (("ce", "Channel\nestimation"), ("rm", "Resource\nmanagement"),
                       ("joint", "Joint\n(CE + RM)")):
        x0, x1 = cols[key]
        _text(ax, (x0 + x1) / 2, 0.795, label, size=_S_HEAD, color=INK, weight="bold")

    _text(ax, 0.065, 0.595, "Learned", size=_S_HEAD, color=INK, weight="bold", rotation=90)
    _text(ax, 0.065, 0.275, "Classical", size=_S_HEAD, color=INK, weight="bold", rotation=90)

    def cell(col, row, *, edge, title, body, fill=None, ls="-", hatch=None,
             title_color=None, title_only=False):
        x0, x1 = cols[col]
        y0, y1 = rows[row]
        _box(ax, x0, y0, x1 - x0, y1 - y0, facecolor=fill or edge, edgecolor=edge,
             alpha=0.12 if fill is None else 0.15, ls=ls, hatch=hatch, lw=1.6,
             learned=(row == "learned" and col != "joint"))
        cx = (x0 + x1) / 2
        if title_only:
            ax.text(cx, (y0 + y1) / 2, title, transform=ax.transAxes, ha="center",
                    va="center", fontsize=_S_HEAD + 1.5, color=title_color or edge,
                    fontweight="bold", linespacing=1.35, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor="none", alpha=0.82))
            return
        _text(ax, cx, y1 - 0.042, title, size=_S_HEAD - 0.5, color=title_color or edge,
              weight="bold")
        _text(ax, cx, y0 + 0.093, body, size=_S_BODY, color="#333333")

    cell("ce", "classical", edge=COL_CE, title="Pilot-based estimators",
         body="LS  ·  DFT denoising\nLMMSE / 2-D Wiener")
    cell("ce", "learned", edge=COL_CE, title="Learned estimators",
         body="CNN denoising\ndeep unfolding\nlearned MMSE")
    cell("rm", "classical", edge=COL_RM, title="Classical schedulers",
         body="PF  ·  round-robin\nWMMSE\nqueue-aware (max-weight)")
    cell("rm", "learned", edge=COL_RM, title="Learned schedulers",
         body="deep RL policies\nlearning to optimise\nGNN-based allocation")
    cell("joint", "classical", edge=SUNWAY_GREY, title="Cross-layer studies",
         body="few studies, e.g.\nRamly et al. (2021):\nMMSE/LS + semi-persistent")
    cell("joint", "learned", edge=COL_GAP, fill=COL_GAP, ls="--", hatch="//",
         title="No reviewed\nstudy", title_color=COL_GAP, body="", title_only=True)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
    _save(fig, output_path, trim=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Figure 2.2: fading scenarios                                   #
# --------------------------------------------------------------------------- #


def _correlated_fading(n: int, rng: np.random.Generator, span: int = 9) -> np.ndarray:
    """Unit-mean-power complex Gaussian process with short-term time correlation."""
    raw = rng.standard_normal(n + span) + 1j * rng.standard_normal(n + span)
    kernel = np.hanning(span)
    kernel /= np.linalg.norm(kernel)
    smoothed = np.convolve(raw, kernel, mode="valid")[:n]
    smoothed /= np.sqrt(np.mean(np.abs(smoothed) ** 2))
    return smoothed


def _seg_hits_box(p0, p1, box) -> bool:
    """True if segment p0-p1 crosses the axis-aligned box (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    for t in np.linspace(0.0, 1.0, 400):
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


def plot_fading_mechanisms(*, output_path: Path) -> None:
    """Side cross-section of a hall: specular paths (Rician) vs blocked (Rayleigh).

    Geometry is computed, not sketched: reflections use the image method, so the
    angle of incidence equals the angle of reflection, and every path is checked
    against the cabinet so the right panel shows only paths that really clear it.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    apply_thesis_rcparams(plt)
    env_fill, env_edge = SEMANTIC["env"]
    link = SEMANTIC["phy"][1]
    W, H = 20.0, 6.0
    ap = (0.6, 4.2)
    dev = (18.0, 2.0)
    machine = (10.0, 14.0, 0.0, 1.0)          # metal machine with a flat top
    cabinet = (8.6, 9.4, 0.0, 5.0)            # tall metal cabinet (right panel)

    # Image-method specular points.
    ceil_img = (ap[0], 2 * H - ap[1])
    t = (ceil_img[1] - H) / (ceil_img[1] - dev[1])
    p_ceil = (ceil_img[0] + t * (dev[0] - ceil_img[0]), H)
    top = machine[3]
    mach_img = (ap[0], 2 * top - ap[1])
    t = (top - mach_img[1]) / (dev[1] - mach_img[1])
    p_mach = (mach_img[0] + t * (dev[0] - mach_img[0]), top)
    assert machine[0] < p_mach[0] < machine[1]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    for k, ax in enumerate(axes):
        blocked = k == 1
        ax.set_xlim(-0.6, W + 0.6)
        ax.set_ylim(-2.3, H + 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        # Hall: floor, ceiling, walls (environment colour).
        ax.add_patch(mpatches.Rectangle((0, 0), W, H, facecolor=env_fill, edgecolor=env_edge,
                                        linewidth=1.4, zorder=0))
        ax.plot([0, W], [H, H], color=env_edge, linewidth=3.0, zorder=1)  # metal roof
        m = machine
        ax.add_patch(mpatches.Rectangle((m[0], m[2]), m[1] - m[0], m[3] - m[2],
                                        facecolor="#c9c9c9", edgecolor="#555555", lw=1.0, zorder=3))
        ax.add_patch(mpatches.Rectangle((dev[0] - 0.15, 0), 0.3, dev[1] - 0.25,
                                        facecolor="#c9c9c9", edgecolor="#555555", lw=0.8, zorder=3))
        ax.add_patch(mpatches.Rectangle((dev[0] - 0.35, dev[1] - 0.25), 0.7, 0.5,
                                        facecolor=link, edgecolor=link, zorder=4))
        ax.add_patch(mpatches.Rectangle((0.0, ap[1] - 0.3), 0.7, 0.6,
                                        facecolor=link, edgecolor=link, zorder=4))
        ax.text(ap[0], -0.35, "Access point", fontsize=9, ha="left", va="top", zorder=6)
        ax.text(dev[0], -0.35, "Device", fontsize=9, ha="center", va="top", zorder=6)
        ax.text((m[0] + m[1]) / 2, m[3] / 2, "Metal machine", fontsize=8, ha="center",
                va="center", color="#333333", zorder=6)
        if not blocked:
            ax.plot([ap[0], dev[0]], [ap[1], dev[1]], color=link, lw=2.6, zorder=5)
            for pt in (p_ceil, p_mach):
                ax.plot([ap[0], pt[0], dev[0]], [ap[1], pt[1], dev[1]], color=link, lw=1.2,
                        ls=(0, (4, 2.5)), zorder=5)
                ax.plot(*pt, marker="o", ms=3, color=link, zorder=6)
            ax.text(9.0, 3.4, "Dominant path present", fontsize=9, color=link, ha="center",
                    va="bottom", rotation=-7, zorder=6)
            ax.text(p_ceil[0], H + 0.2, "Reflection from ceiling", fontsize=9, color=link,
                    ha="center", va="bottom", zorder=6)
            ax.text(p_mach[0], -0.35, "Reflection from machine", fontsize=9, color=link,
                    ha="center", va="top", zorder=6)
            ax.text(W / 2, -1.55, "(a) Rician fading (K > 0)", fontsize=11, ha="center", va="center",
                    fontweight="bold")
        else:
            c = cabinet
            ax.add_patch(mpatches.Rectangle((c[0], c[2]), c[1] - c[0], c[3] - c[2],
                                            facecolor="#9e9e9e", edgecolor="#444444", lw=1.0, zorder=3))
            ax.text((c[0] + c[1]) / 2, -0.35, "Metal cabinet", fontsize=9, ha="center", va="top", zorder=6)
            # Every specular path now crosses the cabinet; draw each up to the cabinet face.
            for start, end in [(ap, dev), (ap, p_mach)]:
                assert _seg_hits_box(start, end, cabinet)
            for path in ([ap, dev], [ap, p_ceil, dev], [ap, p_mach, dev]):
                hit = None
                for a_pt, b_pt in zip(path[:-1], path[1:]):
                    if _seg_hits_box(a_pt, b_pt, cabinet):
                        tt = (c[0] - a_pt[0]) / (b_pt[0] - a_pt[0])
                        hit = (a_pt, (c[0], a_pt[1] + tt * (b_pt[1] - a_pt[1])))
                        break
                assert hit is not None, "a specular path clears the cabinet"
                (sx, sy), (ex, ey) = hit
                pre = path[: path.index(hit[0]) + 1] if hit[0] in path else [ap]
                xs = [q[0] for q in pre] + [ex]
                ys = [q[1] for q in pre] + [ey]
                ax.plot(xs, ys, color="#9a9a9a", lw=1.3, ls=(0, (4, 2.5)), zorder=5)
                ax.plot(ex, ey, marker="x", ms=6, mew=1.6, color="#444444", zorder=6)
            ax.text(4.6, 2.6, "Dominant path blocked", fontsize=9, color="#444444",
                    ha="center", va="top", zorder=6)
            # Diffuse scattering from roof structure beyond the cabinet.
            for sx in (12.5, 14.0, 15.5):
                sp = (sx, H)
                assert not _seg_hits_box(ap, sp, cabinet)
                assert not _seg_hits_box(sp, dev, cabinet)
                ax.plot([ap[0], sp[0], dev[0]], [ap[1], sp[1], dev[1]], color=link, lw=0.9,
                        ls=(0, (1.5, 1.5)), zorder=5)
            ax.text(14.0, -0.35, "Only scattered paths", fontsize=9, color=link, ha="center",
                    va="top", zorder=6)
            ax.text(W / 2, -1.55, "(b) Rayleigh fading (K = 0)", fontsize=11, ha="center",
                    va="center", fontweight="bold")
    fig.tight_layout(h_pad=0.4)
    _save(fig, output_path, trim=True)
    plt.close(fig)


def plot_fading_scenarios(*, output_path: Path) -> None:
    """Rayleigh, Rician, and TR 38.901 UMi fading intuition (P4 anchor, Ch2 SS2.2--2.3).

    (a) outage probability against fade margin for the two statistical models;
    (b) a correlated envelope trace for each, showing how often the link crosses a
    10 dB fade margin; (c) TR 38.901 UMi path loss at 3.5 GHz with the
    factory-cell distance range marked. Rician K matches the canonical runs
    (``rician_k_factor = 1.0``).
    """
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    lab = TABLE_FONT_PT - 4
    ttl = TABLE_FONT_PT - 2
    for ax in axes:
        ax.tick_params(labelsize=lab)
        ax.title.set_fontsize(ttl)

    k_linear = 1.0
    sigma_rayleigh = 1.0 / np.sqrt(2.0)
    sigma_rice = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
    nu_rice = np.sqrt(k_linear / (k_linear + 1.0))

    # (a) Outage probability against fade margin, both models at unit mean power.
    # The CDF (not the PDF) is plotted because the deep-fade tail is the quantity
    # that sets link reliability, and on a linear PDF axis the two models are
    # visually indistinguishable near |h| -> 0.
    ax = axes[0]
    margin_db = np.linspace(0.0, 30.0, 400)
    thresh = 10.0 ** (-margin_db / 20.0)
    out_rayl = stats.rayleigh.cdf(thresh, scale=sigma_rayleigh)
    out_rice = stats.rice.cdf(thresh, nu_rice / sigma_rice, scale=sigma_rice)
    ax.semilogy(margin_db, out_rayl, color=COL_CLASSICAL, lw=2.0, label="Rayleigh")
    ax.semilogy(margin_db, out_rice, color=COL_HYBRID, lw=2.0, label=r"Rician, $K=1$")
    ax.axvline(10.0, color=SUNWAY_GREY, lw=1.0, ls="--")
    # Mark the 10 dB operating point used in panel (b) and state the gap, which
    # at K = 1 is a factor of roughly 1.3 rather than an order of magnitude.
    o_r = float(stats.rayleigh.cdf(10.0 ** (-0.5), scale=sigma_rayleigh))
    o_c = float(stats.rice.cdf(10.0 ** (-0.5), nu_rice / sigma_rice, scale=sigma_rice))
    ax.scatter([10.0, 10.0], [o_r, o_c], s=22,
               color=[COL_CLASSICAL, COL_HYBRID], zorder=5)
    ax.annotate(f"at 10 dB:\n{o_r*100:.1f}% vs {o_c*100:.1f}%\n"
                rf"($\times${o_r/o_c:.2f})",
                xy=(10.0, o_r), xytext=(12.5, 0.18),
                fontsize=lab - 2, color=SUNWAY_GREY, va="bottom", ha="left",
                linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=SUNWAY_GREY, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(r"fade margin $M$ (dB)", fontsize=lab)
    ax.set_ylabel("outage probability", fontsize=lab)
    ax.set_title("(a)  Outage")
    ax.legend(fontsize=lab - 1, loc="lower left", handlelength=1.3, frameon=False,
              borderaxespad=0.2)

    # (b) Correlated envelope traces in dB with a 10 dB fade margin.
    ax = axes[1]
    rng = np.random.default_rng(7)
    n = 220
    t = np.arange(n)
    rayl_db = 20.0 * np.log10(np.abs(_correlated_fading(n, rng)))
    rice_db = 20.0 * np.log10(np.abs(nu_rice + sigma_rice * np.sqrt(2.0) * _correlated_fading(n, rng)))
    ax.axhline(-10.0, color=SUNWAY_GREY, lw=1.0, ls="--")
    ax.plot(t, rayl_db, color=COL_CLASSICAL, lw=1.2, label="Rayleigh")
    ax.plot(t, rice_db, color=COL_HYBRID, lw=1.2, label=r"Rician, $K=1$")
    # Mark the samples that fall below the margin on each trace, and report the
    # fraction so the caption can quote the realised values.
    deep_r = rayl_db < -10.0
    deep_c = rice_db < -10.0
    ax.scatter(t[deep_r], rayl_db[deep_r], s=11, marker="o", color=COL_CLASSICAL,
               edgecolors=COL_GAP, linewidths=0.8, zorder=4)
    ax.scatter(t[deep_c], rice_db[deep_c], s=13, marker="s", color=COL_HYBRID,
               edgecolors=COL_GAP, linewidths=0.8, zorder=4)
    print(f"  fig_fading (b): samples below -10 dB: Rayleigh {deep_r.sum()}/{n}, "
          f"Rician {deep_c.sum()}/{n}")
    ax.set_xlim(0, n)
    ax.set_ylim(-32, 14)
    ax.set_yticks([-30, -20, -10, 0, 10])
    ax.set_xlabel("time (OFDM symbols)", fontsize=lab)
    ax.set_ylabel(r"channel gain $20\log_{10}|h|$ (dB)", fontsize=lab)
    ax.set_title("(b)  Envelopes")
    # Same two series and colours as panel (a), whose legend covers both.

    # (c) TR 38.901 UMi path loss at 3.5 GHz.
    ax = axes[2]
    f_ghz = 3.5
    # TR 38.901 Table 7.4.1-1 defines the UMi path-loss model for
    # 10 m <= d_2D <= 5 km, so the curves start at the 10 m lower bound.
    d_min = 10.0
    d_m = np.linspace(d_min, 40.0, 400)
    # UMi street canyon (h_UT = 1.5 m, so the -0.3*(h_UT - 1.5) NLOS term vanishes).
    pl_los = 32.4 + 21.0 * np.log10(d_m) + 20.0 * np.log10(f_ghz)
    pl_nlos = np.maximum(pl_los, 35.3 * np.log10(d_m) + 22.4 + 21.3 * np.log10(f_ghz))
    # Shade the hall link distances inside the validity range: from 10 m up to
    # the floor diagonal of the 15 m x 15 m reference hall (about 21 m).
    d_hall = 15.0 * np.sqrt(2.0)
    ax.axvspan(d_min, d_hall, color=SUNWAY_NAVY, alpha=0.10, lw=0)
    ax.plot(d_m, pl_los, color=PALETTE[1], lw=2.0, label="LOS")
    ax.plot(d_m, pl_nlos, color=PALETTE[3], lw=2.0, ls="--", label="NLOS")
    for d_mark in (15.0, 40.0):
        i = int(np.argmin(np.abs(d_m - d_mark)))
        print(f"  fig_fading (c): NLOS-LOS gap at {d_mark:.0f} m = "
              f"{pl_nlos[i] - pl_los[i]:.1f} dB")
    ax.set_xlim(d_min, 40.0)
    ax.set_ylim(60, 95)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_xlabel(r"link distance $d_{\mathrm{2D}}$ (m)", fontsize=lab)
    ax.set_ylabel("path loss (dB)", fontsize=lab)
    ax.set_title("(c)  UMi path loss")
    ax.text((d_min + d_hall) / 2, 93.0, "hall\nrange", fontsize=lab - 2,
            color=SUNWAY_NAVY, ha="center", va="top", linespacing=1.05)
    ax.legend(fontsize=lab - 1, loc="lower right", handlelength=1.5,
              borderaxespad=0.2)

    fig.tight_layout(w_pad=1.8)
    _save(fig, output_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Figure 2.3: channel-estimation family trade-off                #
# --------------------------------------------------------------------------- #


def plot_ce_estimator_tradeoff(*, output_path: Path) -> None:
    """Cost against accuracy (matched and mismatched) for channel-estimation families (Ch2 SS2.4).

    Positions are illustrative, drawn from the comparisons reviewed in the text
    (LS/DFT/LMMSE trade-offs; robust-LMMSE mismatch; learned-estimator gains and
    their distribution-shift penalty). The vertical bar on the data-driven point
    is the accuracy lost when the test channel leaves the training distribution.
    """
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Ordinal positions on 0-10 scales (qualitative, not measured).
    # (label, cost, accuracy when assumptions hold, accuracy under mismatch or None,
    #  class, ha, (dx, dy) label offset in points)
    points = [
        ("LS", 1.0, 2.0, None, "classical", "left", (10, 0)),
        ("DFT denoising", 2.6, 4.0, None, "classical", "left", (10, 0)),
        ("robust LMMSE", 4.6, 6.1, None, "classical", "right", (-10, 0)),
        # Learned families under matched conditions sit level with LMMSE with
        # known statistics: the reviewed studies report approaching or matching
        # (ideal) MMSE, not exceeding it.
        ("LMMSE\n(known statistics)", 5.3, 7.8, 5.2, "classical", "right", (-10, 0)),
        ("model-driven\n(deep unfolding)", 7.0, 7.8, 6.7, "learned", "center", (0, -56)),
        ("data-driven\n(CNN / attention)", 8.8, 7.8, 4.6, "learned", "center", (-16, 26)),
    ]
    colors = {"classical": COL_CLASSICAL, "learned": COL_LEARNED}

    for label, cost, acc, acc_mis, cls, ha, (dx, dy) in points:
        if acc_mis is not None:
            ax.annotate("", xy=(cost, acc_mis), xytext=(cost, acc - 0.25),
                        arrowprops=dict(arrowstyle="-|>", color=COL_GAP, lw=1.4),
                        zorder=2)
            ax.scatter(cost, acc_mis, s=70, facecolors="white", edgecolors=colors[cls],
                       linewidths=1.3, zorder=3)
        ax.scatter(cost, acc, s=120, color=colors[cls], edgecolors="white",
                   linewidths=1.0, zorder=4)
        ax.annotate(label, xy=(cost, acc), xytext=(dx, dy), textcoords="offset points",
                    fontsize=FIG_CALLOUT_PT - 4, color=colors[cls], ha=ha, va="center",
                    linespacing=1.1)

    ax.text(8.55, 4.6, "training-\ndistribution\nshift", fontsize=FIG_CALLOUT_PT - 5,
            color=COL_GAP, ha="right", va="center", linespacing=1.05)
    ax.text(5.05, 5.6, "statistics\nmismatch", fontsize=FIG_CALLOUT_PT - 5,
            color=COL_GAP, ha="right", va="center", linespacing=1.05)

    ax.set_xlim(0.2, 10.0)
    ax.set_ylim(0.8, 9.8)
    ax.set_xticks([1.0, 5.0, 9.0])
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_yticks([1.5, 5.0, 9.0])
    ax.set_yticklabels(["low", "medium", "high"])
    ax.set_xlabel("computational cost (qualitative)")
    ax.set_ylabel("estimation accuracy (qualitative)")
    ax.grid(True, which="major")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_CLASSICAL,
                   markeredgecolor="white", markersize=9, label="classical"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_LEARNED,
                   markeredgecolor="white", markersize=9, label="learned"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=SUNWAY_GREY, markersize=7,
                   label="under mismatch"),
    ]
    ax.legend(handles=handles, fontsize=FIG_CALLOUT_PT - 4, loc="lower right", frameon=False)

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Figure 2.4: DRL resource-management loop                       #
# --------------------------------------------------------------------------- #


def plot_drl_rm_loop(*, output_path: Path) -> None:
    """The DRL scheduling loop and where estimator error enters it (Ch2 SS2.5.3)."""
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Policy (top centre)
    _box(ax, 0.36, 0.76, 0.28, 0.17, facecolor=COL_RM, edgecolor=COL_RM, alpha=0.14, lw=1.7,
         learned=True)
    _text(ax, 0.5, 0.875, r"Policy  $\pi_\theta$", size=_S_HEAD, color=COL_RM, weight="bold")
    _text(ax, 0.5, 0.805, "value-based or\nactor–critic network", size=_S_BODY, color="#333333")

    # Radio environment (bottom centre)
    _box(ax, 0.30, 0.04, 0.40, 0.17, facecolor=COL_ENV, edgecolor=COL_ENV,
         alpha=0.11, lw=1.7)
    _text(ax, 0.5, 0.160, "Factory radio environment", size=_S_HEAD, color=COL_ENV,
          weight="bold")
    _text(ax, 0.5, 0.090, "coded OFDM link; Rayleigh, Rician,\nor TR 38.901 channel",
          size=_S_BODY, color="#333333")

    # Channel estimator on the observation path (left) -- first coupling point
    _box(ax, 0.02, 0.40, 0.25, 0.19, facecolor=COL_CE, edgecolor=COL_CE, alpha=0.10, lw=1.6)
    _text(ax, 0.145, 0.540, "Channel estimator", size=_S_HEAD - 0.5, color=COL_CE, weight="bold")
    _text(ax, 0.145, 0.465, "LS, LMMSE, or neural\n" r"$\hat{\mathbf{h}} = \mathbf{h} + \mathbf{e}$",
          size=_S_BODY, color="#333333")
    _arrow(ax, (0.30, 0.125), (0.145, 0.40), color=COL_CE, lw=1.7,
           connectionstyle="arc3,rad=-0.25")
    _text(ax, 0.155, 0.235, "received\npilots", size=_S_NOTE, color=COL_CE, ha="right")
    _arrow(ax, (0.145, 0.59), (0.36, 0.845), color=COL_GAP, lw=1.9,
           connectionstyle="arc3,rad=-0.25")
    _text(ax, 0.02, 0.80, "state $s_t$:\n$\\hat{\\mathbf{h}}$, queue lengths,\nrecent rates",
          size=_S_BODY, color=COL_GAP, ha="left")

    # Reward (centre) -- second coupling point
    _arrow(ax, (0.5, 0.21), (0.5, 0.76), color=COL_HYBRID, lw=1.7)
    ax.text(0.5, 0.555, "reward $r_t$:\nsum rate or $-$BLER", transform=ax.transAxes,
            ha="center", va="center", fontsize=_S_BODY, color=_ink(COL_HYBRID), linespacing=1.35,
            zorder=6, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))
    ax.text(0.5, 0.375, "if computed from $\\hat{\\mathbf{h}}$, a decoding\nfailure goes unseen",
            transform=ax.transAxes, ha="center", va="center", fontsize=_S_NOTE,
            color=COL_GAP, style="italic", linespacing=1.3, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COL_GAP,
                      linewidth=0.9))

    # Action (right)
    _arrow(ax, (0.64, 0.845), (0.70, 0.125), color=COL_RM, lw=1.7,
           connectionstyle="arc3,rad=-0.45")
    _text(ax, 0.98, 0.48, "action $a_t$:\nactive terminals,\ntransmit powers", size=_S_BODY,
          color=COL_RM, ha="right")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save(fig, output_path, trim=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Industry 4.0 to Industry 5.0                                   #
# --------------------------------------------------------------------------- #


def plot_industry50(*, output_path: Path) -> None:
    """Industry 4.0 base, the three Industry 5.0 pillars, and the wireless need both share."""
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Industry 4.0: technology-driven base (left)
    x4, w4, y_top, y_bot = 0.02, 0.28, 0.97, 0.30
    _box(ax, x4, y_bot, w4, y_top - y_bot, facecolor=COL_CLASSICAL, edgecolor=COL_CLASSICAL,
         alpha=0.05, lw=1.2)
    _text(ax, x4 + w4 / 2, y_top - 0.045, "Industry 4.0", size=_S_HEAD + 1,
          color=COL_CLASSICAL, weight="bold")
    _text(ax, x4 + w4 / 2, y_top - 0.105, "technology-driven", size=_S_NOTE,
          color="#555555", style="italic")
    items4 = ["cyber-physical\nsystems", "IoT and wireless\nsensing", "cloud and data\nanalytics",
              "automation for\nefficiency"]
    bh, gap = 0.105, 0.025
    y = y_top - 0.165 - bh
    for label in items4:
        _box(ax, x4 + 0.025, y, w4 - 0.05, bh, facecolor=COL_CLASSICAL, edgecolor=COL_CLASSICAL,
             alpha=0.14, lw=1.0)
        _text(ax, x4 + w4 / 2, y + bh / 2, label, size=_S_BODY, color="#2a2a2a")
        y -= bh + gap

    # transition arrow
    x5, w5 = 0.45, 0.53
    _arrow(ax, (x4 + w4 + 0.012, 0.60), (x5 - 0.008, 0.60), color=SUNWAY_GREY, lw=1.8)
    _text(ax, (x4 + w4 + x5) / 2, 0.69, "keeps the\ninfrastructure,\nadds values",
          size=_S_NOTE, color="#555555", style="italic")

    # Industry 5.0: value-driven, three pillars (right)
    _box(ax, x5, y_bot, w5, y_top - y_bot, facecolor=COL_CLASSICAL, edgecolor=COL_CLASSICAL,
         alpha=0.05, lw=1.2)
    _text(ax, x5 + w5 / 2, y_top - 0.045, "Industry 5.0", size=_S_HEAD + 1,
          color=INK, weight="bold")
    _text(ax, x5 + w5 / 2, y_top - 0.105, "value-driven, built on Industry 4.0",
          size=_S_NOTE, color="#555555", style="italic")
    pillars = [
        ("Human-centric", "operator at the\ncentre; robot as\ncollaborator",
         "safe cobot motion\nneeds timely\ncommands", COL_CLASSICAL),
        ("Sustainable", "lower energy\nand material\nuse",
         "fewer cables,\nreusable lines", COL_CLASSICAL),
        ("Resilient", "robust to\ndisruption; fast\nreconfiguration",
         "links that hold\nunder blockage\nand fading", COL_CLASSICAL),
    ]
    pw, pg = 0.155, 0.02
    px = x5 + (w5 - 3 * pw - 2 * pg) / 2
    p_bot, p_top = y_bot + 0.035, y_top - 0.165
    for name, desc, link_role, col in pillars:
        _box(ax, px, p_bot, pw, p_top - p_bot, facecolor=col, edgecolor=col, alpha=0.13, lw=1.2)
        _text(ax, px + pw / 2, p_top - 0.045, name, size=_S_BODY + 0.5, color=INK, weight="bold")
        _text(ax, px + pw / 2, p_top - 0.095, desc, size=_S_NOTE + 0.3, color="#2a2a2a", va="top")
        ax.plot([px + 0.02, px + pw - 0.02], [p_bot + 0.145] * 2, color="#a8a8a8", lw=0.8,
                alpha=0.6, transform=ax.transAxes, zorder=4)
        _text(ax, px + pw / 2, p_bot + 0.072, link_role, size=_S_NOTE, color="#555555",
              style="italic")
        px += pw + pg

    # shared wireless requirement (bottom band)
    _box(ax, 0.02, 0.02, 0.96, 0.20, facecolor=COL_GAP, edgecolor=COL_GAP, alpha=0.07, lw=1.6)
    _text(ax, 0.5, 0.165, "Shared requirement: wireless links inside the control and safety path",
          size=_S_HEAD, color=COL_GAP, weight="bold")
    _text(ax, 0.5, 0.08,
          "closed-loop control, mobile robots and cobot\u2013operator messages need low latency "
          "and high reliability\n\u2192 set by channel estimation and resource management "
          "on the factory link",
          size=_S_NOTE + 0.3, color="#2a2a2a")
    _arrow(ax, (x4 + w4 / 2, y_bot - 0.005), (x4 + w4 / 2, 0.225), color=SUNWAY_GREY, lw=1.3)
    _arrow(ax, (x5 + w5 / 2, y_bot - 0.005), (x5 + w5 / 2, 0.225), color=SUNWAY_GREY, lw=1.3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save(fig, output_path, trim=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chapter 2 -- Figure 2.5: evaluation-coupling patterns                       #
# --------------------------------------------------------------------------- #


def _flow(ax: Any, x0: float, y: float, width: float, labels: Sequence[str],
          color: str, *, box_h: float = 0.115) -> list[float]:
    """Left-to-right chain of boxes joined by arrows; returns box centre x-values."""
    n = len(labels)
    gap = 0.030
    bw = (width - gap * (n - 1)) / n
    centers = []
    for i, label in enumerate(labels):
        x = x0 + i * (bw + gap)
        _box(ax, x, y, bw, box_h, facecolor=color, edgecolor=color, alpha=0.16, lw=1.2)
        _text(ax, x + bw / 2, y + box_h / 2, label, size=_S_BODY, color="#2a2a2a")
        centers.append(x + bw / 2)
        if i > 0:
            _arrow(ax, (x - gap * 0.9, y + box_h / 2), (x - gap * 0.1, y + box_h / 2),
                   color=SUNWAY_GREY, lw=1.2)
    return centers


def plot_lr_eval_coupling(*, output_path: Path) -> None:
    """PHY-only, MAC-only, and coupled evaluation patterns in the CE/RM literature (Ch2 SS2.7)."""
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lanes = [
        {
            "y": 0.660, "color": COL_CE,
            "title": "PHY-only estimator studies",
            "flow": ["received\npilots", "estimator", "equalise\n+ decode", "BER / MSE"],
            "foot": "estimate stops here — no scheduler in the loop",
        },
        {
            "y": 0.370, "color": COL_RM,
            "title": "MAC-only scheduler studies",
            "flow": ["perfect, delayed,\nor quantised CSI", "scheduler", "rate /\nfairness"],
            "foot": "estimator error never reaches the scheduler",
        },
        {
            "y": 0.085, "color": COL_GAP,
            "title": "Coupled CE + RM  (rare)",
            "flow": ["shared\nchannel draw", "estimator", r"$\hat{\mathbf{h}}$ + error",
                     "scheduler", "end-to-end\nBER + rate"],
            "foot": "shared realisations, estimated CSI — seldom on factory fading",
        },
    ]

    for lane in lanes:
        y = lane["y"]
        emph = lane["color"] == COL_GAP
        _box(ax, 0.035, y - 0.052, 0.93, 0.205, facecolor=lane["color"], edgecolor=lane["color"],
             alpha=0.09 if emph else 0.05, lw=1.6 if emph else 1.0)
        _text(ax, 0.06, y + 0.120, lane["title"], size=_S_HEAD, color=lane["color"],
              weight="bold", ha="left")
        _flow(ax, 0.065, y, 0.87, lane["flow"], lane["color"], box_h=0.100)
        _text(ax, 0.52, y - 0.030, lane["foot"], size=_S_NOTE, color="#555555", style="italic")

    # the estimator -> scheduler link the first two lanes never make
    _arrow(ax, (0.30, 0.602), (0.30, 0.532), color=COL_GAP, lw=1.6, ls=(0, (4, 3)))
    _text(ax, 0.315, 0.567, "missing link", size=_S_NOTE, color=COL_GAP, style="italic", ha="left")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03)
    _save(fig, output_path, trim=True)
    plt.close(fig)
