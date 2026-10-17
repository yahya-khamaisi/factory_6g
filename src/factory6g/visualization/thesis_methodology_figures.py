"""Chapter 3 methodology figures rendered with matplotlib.

Holds the scaled plan view of the canonical small-factory deployment profile
(``fig_factory_plan.png``), the uplink OFDM resource grid with its Kronecker
pilot pattern (``fig_resource_grid.png``) and the antenna configuration
(``fig_antenna_array.png``).  Machine placement reuses the same
``numpy`` generator and seed as
:func:`factory6g.visualization.factory_visualizer._build_scene_xml`, so the
published figure matches the layout the ray-tracing backend would build from
the same configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .thesis_plot_style import apply_thesis_rcparams

# Canonical small-factory profile (config/config.json, factory size preset "s").
ROOM_DIMENSIONS: tuple[float, float, float] = (15.0, 15.0, 5.0)
NUM_MACHINES: int = 5
MACHINE_SIZE_RANGE: tuple[tuple[float, float], ...] = ((0.5, 2.0), (0.5, 2.0), (0.5, 1.5))
ROOM_PADDING: float = 2.0
RX_BOUNDARY_PADDING: float = 1.0
NUM_UT: int = 4
MACHINE_SEED: int = 42
# Chosen so the four terminals occupy distinct quadrants and clear the machine
# footprints; any seed is admissible, this one simply reads well in print.
UT_SEED: int = 219

_ROOM_EDGE = "#1f2328"
_MACHINE_FILL = "#e6e6e6"
_MACHINE_EDGE = "#6b6b6b"
_BS_COLOR = "#1f2328"  # ink: terminal colours mark identity, the BS stays neutral
_UT_COLOR = "#cc6677"

# Uplink OFDM numerology (config/config.json "system"); mirrors the Sionna
# ResourceGrid built in factory6g.models.model with pilot_pattern="kronecker".
NUM_OFDM_SYMBOLS: int = 14
FFT_SIZE: int = 128
PILOT_OFDM_SYMBOLS: tuple[int, ...] = (2, 11)
CARRIER_FREQUENCY_HZ: float = 3.5e9
NUM_BS_ELEMENTS: int = 4  # 8 ports = 4 cross-polarised elements x 2 polarisations
# Heights used by the scene builder (factory_visualizer: BS at ceiling minus
# transceiver.tx_height_offset, terminals at transceiver.rx_height).
BS_HEIGHT_M: float = ROOM_DIMENSIONS[2] - 1.0
UT_HEIGHT_M: float = 1.0
ANTENNA_SPACING_WAVELENGTHS: float = 0.5


def sample_machine_layout(
    room_dimensions: Sequence[float] = ROOM_DIMENSIONS,
    *,
    num_machines: int = NUM_MACHINES,
    size_range: Sequence[Sequence[float]] = MACHINE_SIZE_RANGE,
    padding: float = ROOM_PADDING,
    seed: int = MACHINE_SEED,
) -> list[dict[str, float]]:
    """Return machine footprints in metres, centred on the hall origin."""
    rng = np.random.default_rng(seed)
    room_len, room_width, _ = (float(v) for v in room_dimensions)
    layout: list[dict[str, float]] = []
    for _ in range(int(num_machines)):
        sx = float(rng.uniform(size_range[0][0], size_range[0][1]))
        sy = float(rng.uniform(size_range[1][0], size_range[1][1]))
        sz = float(rng.uniform(size_range[2][0], size_range[2][1]))
        px = float(rng.uniform(-room_len / 2 + padding, room_len / 2 - padding))
        py = float(rng.uniform(-room_width / 2 + padding, room_width / 2 - padding))
        layout.append({"x": px, "y": py, "sx": sx, "sy": sy, "sz": sz})
    return layout


def sample_ut_positions(
    room_dimensions: Sequence[float] = ROOM_DIMENSIONS,
    *,
    num_ut: int = NUM_UT,
    padding: float = RX_BOUNDARY_PADDING,
    seed: int = UT_SEED,
) -> list[tuple[float, float]]:
    """Return user-terminal positions in metres, centred on the hall origin."""
    rng = np.random.default_rng(seed)
    room_len, room_width, _ = (float(v) for v in room_dimensions)
    return [
        (
            float(rng.uniform(-room_len / 2 + padding, room_len / 2 - padding)),
            float(rng.uniform(-room_width / 2 + padding, room_width / 2 - padding)),
        )
        for _ in range(int(num_ut))
    ]


def _label_offset(x: float, y: float) -> tuple[int, int]:
    """Push a terminal label away from the hall centre so it never sits on a link."""
    return (10 if x >= 0 else -10, 10 if y >= 0 else -16)


def render_factory_plan(output_path: str | Path) -> Path:
    """Render the scaled plan view of the canonical small-factory profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)

    room_len, room_width, room_height = ROOM_DIMENSIONS
    machines = sample_machine_layout()
    uts = sample_ut_positions()

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.add_patch(
        mpatches.Rectangle(
            (-room_len / 2, -room_width / 2),
            room_len,
            room_width,
            linewidth=1.8,
            edgecolor=_ROOM_EDGE,
            facecolor="white",
            zorder=1,
        )
    )

    for index, machine in enumerate(machines):
        ax.add_patch(
            mpatches.Rectangle(
                (machine["x"] - machine["sx"] / 2, machine["y"] - machine["sy"] / 2),
                machine["sx"],
                machine["sy"],
                linewidth=1.0,
                edgecolor=_MACHINE_EDGE,
                facecolor=_MACHINE_FILL,
                zorder=2,
                label="Metal machine" if index == 0 else None,
            )
        )

    bs_xy = (0.0, 0.0)
    # Each terminal keeps one colour across the plan, the 3D view and the resource grid.
    ut_colors = _terminal_colors(len(uts))
    for index, (ux, uy) in enumerate(uts):
        ax.plot(
            [ux, bs_xy[0]],
            [uy, bs_xy[1]],
            color=ut_colors[index],
            linewidth=0.9,
            linestyle="--",
            alpha=0.55,
            zorder=3,
        )
        ax.scatter(
            [ux],
            [uy],
            s=70,
            marker="o",
            color=ut_colors[index],
            zorder=5,
            label="User terminal (1 antenna)" if index == 0 else None,
        )
        dx, dy = _label_offset(ux, uy)
        ax.annotate(
            f"UT{index + 1}",
            (ux, uy),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            color=ut_colors[index],
            fontsize=11,
            zorder=6,
        )

    ax.scatter(
        [bs_xy[0]],
        [bs_xy[1]],
        s=170,
        marker="^",
        color=_BS_COLOR,
        zorder=7,
        label="Base station (8 antenna ports)",
    )

    ax.set_xlim(-room_len / 2 - 1.2, room_len / 2 + 1.2)
    ax.set_ylim(-room_width / 2 - 1.2, room_width / 2 + 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # Major ticks every 5 m keep the labels legible at print width; minor ticks
    # every 2.5 m mark the hall edges and carry the grid.
    ax.set_xticks(np.arange(-5.0, room_len / 2 + 0.1, 5.0))
    ax.set_yticks(np.arange(-5.0, room_width / 2 + 0.1, 5.0))
    ax.set_xticks(np.arange(-room_len / 2, room_len / 2 + 0.1, 2.5), minor=True)
    ax.set_yticks(np.arange(-room_width / 2, room_width / 2 + 0.1, 2.5), minor=True)
    ax.grid(True, which="major")
    # No in-image title: the LaTeX caption describes the hall.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def kronecker_pilot_owner(
    num_ut: int = NUM_UT,
    num_ofdm_symbols: int = NUM_OFDM_SYMBOLS,
    fft_size: int = FFT_SIZE,
    pilot_symbols: Sequence[int] = PILOT_OFDM_SYMBOLS,
) -> np.ndarray:
    """Return a [symbol, subcarrier] map: -1 for data, u for terminal u's pilot.

    On each pilot OFDM symbol, terminal u sends a non-zero pilot on subcarriers
    k with k mod U == u and is silent on the other pilot positions; every
    other resource element carries data from all active terminals. This is the
    Sionna KroneckerPilotPattern for U single-stream transmitters (checked
    against Sionna in tests/test_methodology_figures.py).
    """
    owner = np.full((num_ofdm_symbols, fft_size), -1, dtype=int)
    for sym in pilot_symbols:
        owner[sym, :] = np.arange(fft_size) % num_ut
    return owner


def _terminal_colors(num_ut: int) -> list[str]:
    from .thesis_plot_style import PALETTE

    return [PALETTE[i % len(PALETTE)] for i in range(num_ut)]


def render_resource_grid(output_path: str | Path, *, zoom_subcarriers: int = 8) -> Path:
    """Render the uplink resource grid: full-grid overview plus a pilot close-up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    apply_thesis_rcparams(plt)

    owner = kronecker_pilot_owner()
    colors = _terminal_colors(NUM_UT)
    data_color = "#eeeeee"
    # Index 0 = data, 1..U = terminal pilots.
    cmap = ListedColormap([data_color] + colors)
    image = owner.T + 1  # [subcarrier, symbol]

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(6.4, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]}, layout="constrained"
    )

    # (a) Whole grid: 14 OFDM symbols by 128 subcarriers.
    ax_full.imshow(image, cmap=cmap, vmin=0, vmax=NUM_UT, aspect="auto",
                   interpolation="nearest", origin="lower",
                   extent=(-0.5, NUM_OFDM_SYMBOLS - 0.5, -0.5, FFT_SIZE - 0.5))
    ax_full.grid(False)
    ax_full.set_xticks(range(0, NUM_OFDM_SYMBOLS, 1))
    ax_full.set_xticklabels([str(i) if i in (0, *PILOT_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS - 1) else ""
                             for i in range(NUM_OFDM_SYMBOLS)])
    ax_full.set_yticks([0, 32, 64, 96, 127])
    ax_full.set_xlabel("OFDM symbol")
    ax_full.set_ylabel("Subcarrier")
    ax_full.set_title("(a) Full slot", fontsize=13)
    ax_full.add_patch(mpatches.Rectangle(
        (-0.5, -0.5), NUM_OFDM_SYMBOLS, zoom_subcarriers, fill=False,
        edgecolor="#1f2328", linewidth=1.4, zorder=3))

    # (b) Close-up of the lowest subcarriers, one cell per resource element.
    zoom = image[:zoom_subcarriers, :]
    ax_zoom.imshow(zoom, cmap=cmap, vmin=0, vmax=NUM_UT, aspect="auto",
                   interpolation="nearest", origin="lower",
                   extent=(-0.5, NUM_OFDM_SYMBOLS - 0.5, -0.5, zoom_subcarriers - 0.5))
    ax_zoom.grid(False)
    for x in np.arange(-0.5, NUM_OFDM_SYMBOLS, 1.0):
        ax_zoom.axvline(x, color="white", linewidth=1.2)
    for y in np.arange(-0.5, zoom_subcarriers, 1.0):
        ax_zoom.axhline(y, color="white", linewidth=1.2)
    for sym in PILOT_OFDM_SYMBOLS:
        for k in range(zoom_subcarriers):
            u = int(owner[sym, k])
            ax_zoom.text(sym, k, str(u + 1), ha="center", va="center",
                         color="white", fontsize=10, fontweight="bold")
    ax_zoom.set_xticks(range(NUM_OFDM_SYMBOLS))
    ax_zoom.set_yticks(range(zoom_subcarriers))
    ax_zoom.tick_params(length=0)
    # Fourteen two-digit labels need a smaller size to stay apart.
    ax_zoom.tick_params(axis="x", labelsize=10)
    for side in ("left", "bottom"):
        ax_zoom.spines[side].set_visible(False)
    ax_zoom.set_xlabel("OFDM symbol")
    ax_zoom.set_ylabel("Subcarrier")
    ax_zoom.set_title(f"(b) Subcarriers 0\u2013{zoom_subcarriers - 1}", fontsize=13)

    handles = [mpatches.Patch(facecolor=c, edgecolor="none", label=f"Pilot, UT{u + 1}")
               for u, c in enumerate(colors)]
    handles.append(mpatches.Patch(facecolor=data_color, edgecolor="#bdbdbd",
                                  label="Data, all active UTs"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_antenna_array(output_path: str | Path) -> Path:
    """Render the BS cross-polarised linear array (8 ports) and the UT element."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .thesis_plot_style import INK, MUTED, PALETTE

    apply_thesis_rcparams(plt)

    wavelength_cm = 299_792_458.0 / CARRIER_FREQUENCY_HZ * 100.0
    spacing_cm = ANTENNA_SPACING_WAVELENGTHS * wavelength_cm
    plus_color, minus_color = PALETTE[0], PALETTE[1]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.set_aspect("equal")
    ax.axis("off")

    # Base-station array: elements at 0, d, 2d, 3d along x (unit = spacing d).
    half = 0.32
    for i in range(NUM_BS_ELEMENTS):
        x = float(i)
        ax.add_patch(plt.Rectangle((x - half, -half), 2 * half, 2 * half,
                                   fill=False, edgecolor=INK, linewidth=1.0))
        ax.plot([x - 0.24, x + 0.24], [-0.24, 0.24], color=plus_color, linewidth=2.4,
                solid_capstyle="round")
        ax.plot([x - 0.24, x + 0.24], [0.24, -0.24], color=minus_color, linewidth=2.4,
                solid_capstyle="round")
    # Element-spacing dimension between the first two elements.
    ax.annotate("", xy=(1.0, -0.62), xytext=(0.0, -0.62),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.9))
    for x in (0.0, 1.0):
        ax.plot([x, x], [-0.36, -0.68], color=MUTED, linewidth=0.6)
    ax.text(0.5, -0.82, rf"$d=\lambda/2\approx{spacing_cm:.1f}$ cm",
            ha="center", va="top", fontsize=11)
    ax.text((NUM_BS_ELEMENTS - 1) / 2, 0.55,
            f"Base station: {NUM_BS_ELEMENTS} cross-polarised elements = "
            f"{2 * NUM_BS_ELEMENTS} ports", ha="center", va="bottom", fontsize=11)

    # User terminal: one vertically polarised element.
    ux = NUM_BS_ELEMENTS + 1.3
    ax.add_patch(plt.Rectangle((ux - half, -half), 2 * half, 2 * half,
                               fill=False, edgecolor=INK, linewidth=1.0))
    ax.plot([ux, ux], [-0.26, 0.26], color=INK, linewidth=2.4, solid_capstyle="round")
    ax.text(ux, 0.55, "Each UT:\n1 vertical element", ha="center", va="bottom",
            fontsize=11, linespacing=1.1)

    ax.set_xlim(-0.6, ux + 0.9)
    ax.set_ylim(-1.15, 1.25)
    handles = [
        Line2D([], [], color=plus_color, linewidth=2.4, label=r"$+45^{\circ}$ port"),
        Line2D([], [], color=minus_color, linewidth=2.4, label=r"$-45^{\circ}$ port"),
        Line2D([], [], color=INK, linewidth=2.4, label="Vertical port"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              frameon=False, fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_factory_hall_3d(output_path: str | Path) -> Path:
    """Render an isometric 3D view of the canonical hall from the plan's own layout.

    Uses sample_machine_layout() and sample_ut_positions(), so every position is
    the one drawn in fig_factory_plan.png; heights follow the scene builder.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from .thesis_plot_style import INK, MUTED

    apply_thesis_rcparams(plt)

    room_len, room_width, room_height = ROOM_DIMENSIONS
    hx, hy = room_len / 2, room_width / 2
    machines = sample_machine_layout()
    uts = sample_ut_positions()
    ut_colors = _terminal_colors(len(uts))

    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # Chosen by maximising on-screen separation between the hanging base station,
    # the terminals and the machines over a grid of view angles.
    ax.view_init(elev=40, azim=-25)
    ax.set_box_aspect((room_len, room_width, room_height), zoom=0.92)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    # Draw strictly by zorder: floor < walls < solids < links < labels. Solids share
    # one collection, which matplotlib depth-sorts polygon by polygon.
    ax.computed_zorder = False

    def box_faces(x0, y0, z0, dx, dy, dz):
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        return [
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front (-y)
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right (+x)
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        ]

    # Floor, faint 2.5 m grid, and the two far walls (near walls cut away).
    ax.add_collection3d(Poly3DCollection(
        [[(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]],
        facecolor="#f6f6f6", edgecolor=INK, linewidth=1.0, zorder=0))
    grid = [[(v, -hy, 0), (v, hy, 0)] for v in np.arange(-hx + 2.5, hx, 2.5)]
    grid += [[(-hx, v, 0), (hx, v, 0)] for v in np.arange(-hy + 2.5, hy, 2.5)]
    ax.add_collection3d(Line3DCollection(grid, colors="#dedede", linewidths=0.6, zorder=1))
    walls = [
        [(-hx, hy, 0), (hx, hy, 0), (hx, hy, room_height), (-hx, hy, room_height)],
        [(-hx, -hy, 0), (-hx, hy, 0), (-hx, hy, room_height), (-hx, -hy, room_height)],
    ]
    ax.add_collection3d(Poly3DCollection(walls, facecolor="#eeeeee", edgecolor=MUTED,
                                         linewidth=0.8, zorder=1))

    polys: list = []
    face_colors: list[str] = []
    edge_colors: list[str] = []
    # Machines: top face lightest, sides shaded, so the blocks read as solids.
    shades = ["#dcdcdc", "#bcbcbc", "#a8a8a8", "#bcbcbc", "#a8a8a8"]
    for m in machines:
        polys += box_faces(m["x"] - m["sx"] / 2, m["y"] - m["sy"] / 2, 0.0,
                           m["sx"], m["sy"], m["sz"])
        face_colors += shades
        edge_colors += ["#7a7a7a"] * 5
    # Base station: a horizontal four-element panel hanging from the ceiling.
    bs_z = BS_HEIGHT_M
    polys += box_faces(-1.0, -0.22, bs_z - 0.15, 2.0, 0.44, 0.3)
    face_colors += [INK] * 5
    edge_colors += [INK] * 5
    # Terminals: small boxes in their own colour on stands.
    for i, (ux, uy) in enumerate(uts):
        polys += box_faces(ux - 0.35, uy - 0.35, UT_HEIGHT_M - 0.25, 0.7, 0.7, 0.45)
        face_colors += [ut_colors[i]] * 5
        edge_colors += [ut_colors[i]] * 5
    ax.add_collection3d(Poly3DCollection(polys, facecolors=face_colors,
                                         edgecolors=edge_colors, linewidths=0.5, zorder=2))

    ax.plot([0, 0], [0, 0], [bs_z + 0.12, room_height], color=INK, linewidth=1.6, zorder=3)
    # Dotted drop line and floor mark: the panel hangs above the hall centre.
    ax.plot([0, 0], [0, 0], [0, bs_z - 0.12], color=MUTED, linewidth=0.9,
            linestyle=(0, (1, 2)), zorder=3)
    ax.plot([-0.35, 0.35], [0, 0], [0, 0], color=MUTED, linewidth=1.0, zorder=3)
    ax.plot([0, 0], [-0.35, 0.35], [0, 0], color=MUTED, linewidth=1.0, zorder=3)
    for i, (ux, uy) in enumerate(uts):
        c = ut_colors[i]
        ax.plot([ux, ux], [uy, uy], [0, UT_HEIGHT_M - 0.2], color=MUTED, linewidth=1.2,
                zorder=3)
        ant_top = UT_HEIGHT_M + 0.55
        ax.plot([ux, ux], [uy, uy], [UT_HEIGHT_M + 0.15, ant_top], color=c, linewidth=1.6,
                zorder=3)
        ax.plot([ux, 0], [uy, 0], [ant_top, bs_z - 0.12], color=c, linewidth=1.1,
                linestyle=(0, (4, 3)), zorder=3)
        # Label placement keeps each name off its own uplink: left-half terminals
        # on their left, far-right terminals below-right, the rest above-right.
        if ux < 0:
            pos, ha = (ux - 0.5, uy, ant_top + 0.3), "right"
        elif ux > 3.5:
            pos, ha = (ux + 0.6, uy, 0.1), "left"
        else:
            pos, ha = (ux + 0.5, uy, ant_top + 0.3), "left"
        ax.text(*pos, f"UT{i + 1}", color=c, fontsize=13, fontweight="bold", zorder=5, ha=ha)

    # Base-station label sits in clear floor space, joined to the panel by a leader.
    from mpl_toolkits.mplot3d import proj3d

    bx, by, _ = proj3d.proj_transform(0.0, 0.0, bs_z, ax.get_proj())
    ax.annotate("Base station (8 ports)", xy=(bx, by), xycoords="data",
                xytext=(75, 14), textcoords="offset points", ha="left", va="center",
                fontsize=12.5, color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkB=4))
    # Label the machine nearest the viewer, on the floor just in front of it.
    front = min(machines, key=lambda m: m["y"])
    ax.text(front["x"], front["y"] - front["sy"] / 2 - 1.1, 0, "Metal machine",
            color=MUTED, fontsize=12, ha="center", va="top", zorder=5)

    # Dimensions on the near floor edges and the far-left vertical edge.
    ax.text(0, -hy - 1.6, 0, f"{room_len:.0f} m", ha="center", va="top", fontsize=13,
            color=INK, zorder=5)
    ax.text(hx + 1.4, 0, 0, f"{room_width:.0f} m", ha="left", va="center", fontsize=13,
            color=INK, zorder=5)
    ax.text(-hx - 0.6, hy, room_height / 2, f"{room_height:.0f} m", ha="right",
            va="center", fontsize=13, color=INK, zorder=5)

    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)
    ax.set_zlim(0, room_height)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    _trim_white_margins(output_path)
    return output_path


def _trim_white_margins(path: Path, pad: int = 30) -> None:
    """Crop the empty band a 3D axes box leaves around the drawing."""
    from PIL import Image

    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = np.where((255 - rgb).sum(axis=2) > 24)
    if ink[0].size == 0:
        return
    top, bottom = max(int(ink[0].min()) - pad, 0), min(int(ink[0].max()) + pad, rgb.shape[0])
    left, right = max(int(ink[1].min()) - pad, 0), min(int(ink[1].max()) + pad, rgb.shape[1])
    img.crop((left, top, right, bottom)).save(path, dpi=img.info.get("dpi", (300, 300)))


def render_all(output_dir: str | Path) -> dict[str, str]:
    """Render every Chapter 3 matplotlib figure into ``output_dir``."""
    output_dir = Path(output_dir)
    produced: dict[str, str] = {}
    path = render_factory_plan(output_dir / "fig_factory_plan.png")
    produced["fig_factory_plan"] = str(path)
    path = render_factory_hall_3d(output_dir / "fig_factory_hall_3d.png")
    produced["fig_factory_hall_3d"] = str(path)
    path = render_resource_grid(output_dir / "fig_resource_grid.png")
    produced["fig_resource_grid"] = str(path)
    path = render_antenna_array(output_dir / "fig_antenna_array.png")
    produced["fig_antenna_array"] = str(path)
    return produced


__all__ = [
    "kronecker_pilot_owner",
    "render_all",
    "render_antenna_array",
    "render_factory_hall_3d",
    "render_factory_plan",
    "render_resource_grid",
    "sample_machine_layout",
    "sample_ut_positions",
]
