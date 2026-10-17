"""Matplotlib style for PhD thesis figures (Sunway / IEEE, reproducible exports).

Academic minimal theme (Oct 2026), shared with the draw.io diagrams in
``scripts/tools/build_thesis_drawio.py``: Times-compatible serif text (STIX in
the container), ink axes with no top/right spines, a faint major-only grid, no
legend frames, and a colour-blind-safe series palette that leads with Sunway
navy. Series stay distinguishable in greyscale through their markers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Printed at \linewidth (~145 mm). Aug 2026: larger labels for Ch1 engagement figures
# (supervisor: figure text too small); wrap earlier so callouts stay readable.
TABLE_FONT_PT = 15
FIG_TITLE_PT = 16
FIG_CALLOUT_PT = 14
# Per-line character caps for matplotlib annotations (Figure label line breaking).
FIG_LABEL_MAX_CHARS = 22
FIG_NOTE_MAX_CHARS = 40

_SEMANTIC_BREAKS = (" · ", " → ", "; ", " — ", " – ", ", ")


def _wrap_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]
    for sep in _SEMANTIC_BREAKS:
        if sep not in paragraph:
            continue
        parts = paragraph.split(sep)
        lines: list[str] = []
        current = parts[0].strip()
        for part in parts[1:]:
            candidate = f"{current}{sep}{part}" if current else part.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = part.strip()
        if current:
            lines.append(current)
        if len(lines) > 1:
            return lines
    words = paragraph.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [paragraph]


def wrap_figure_text(text: str, *, max_chars: int = FIG_LABEL_MAX_CHARS) -> str:
    """Return text with \\n breaks for matplotlib (Figure label line breaking)."""
    lines_out: list[str] = []
    for paragraph in text.replace("<br>", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines_out.extend(_wrap_paragraph(paragraph, max_chars=max_chars))
    return "\n".join(lines_out)


# Thesis body figures: slightly under linewidth (~5.7 in) so fonts stay near body size in print.
THESIS_FIGSIZE = (5.8, 3.8)
# Sweeps carry a legend outside the axes; the extra width keeps the plotting
# area the same size as a plain THESIS_FIGSIZE figure.
THESIS_FIGSIZE_LEGEND = (7.6, 3.8)
THESIS_DPI = 300
EBNO_XLABEL = r"$E_b/N_0$ (dB)"

# Canonical legend order (classical → learned). Unknown keys sort after, alphabetically.
ESTIMATOR_ORDER: tuple[str, ...] = (
    "ls",
    "dft",
    "lmmse",
    "adaptive",
    "neural",
    "pso",
)
RESOURCE_MANAGER_ORDER: tuple[str, ...] = (
    "static",
    "round_robin",
    "max_throughput",
    "pf",
    "wmmse",
    "queue_aware",
    "drl",
    "reliability_drl",
    "cnn",
)

INK = "#1f2328"
MUTED = "#5f6368"
SUNWAY_NAVY = "#233369"
NAVY_TINT = "#eef0f6"
GRID = "#e4e4e4"

# Colour-blind-safe series palette (after Paul Tol's "muted" scheme, with
# Sunway navy in place of indigo). Adjacent entries differ in hue and
# lightness so neighbouring legend entries never look alike.
PALETTE: tuple[str, ...] = (
    "#233369",  # navy
    "#cc6677",  # rose
    "#117733",  # green
    "#aa4499",  # purple
    "#3f8fbf",  # blue
    "#999933",  # olive
    "#882255",  # wine
    "#44aa99",  # teal
    "#6b6b6b",  # grey
    "#ddaa33",  # ochre
)

# Semantic diagram colours (fill, stroke), shared with scripts/tools/build_thesis_drawio.py
# (keep the two in step). Colour means the same thing in every schematic:
# env = environment and inputs, phy = physical layer, mac = scheduling,
# learned = learned component, coupling = PHY-MAC coupling, output = metrics,
# process = neutral procedure. Okabe-Ito based, colour-vision-deficiency safe.
SEMANTIC: dict[str, tuple[str, str]] = {
    "env": ("#f1eee9", "#7a6c5d"),
    "phy": ("#e2eef8", "#0b5e9c"),
    "mac": ("#fdefd3", "#a66900"),
    "learned": ("#f6e8f1", "#a2508a"),
    "coupling": ("#fbe4d8", "#c0510a"),
    "output": ("#dcf1e9", "#007a5a"),
    "process": ("#f5f5f5", "#5f6368"),
}

MARKERS: tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X", "*", "+", "h")


def _canonical_order(stage_hint: str | None) -> tuple[str, ...]:
    if stage_hint in {"estimators", "estimator"}:
        return ESTIMATOR_ORDER
    if stage_hint in {"resource_managers", "resource_manager"}:
        return RESOURCE_MANAGER_ORDER
    merged = list(ESTIMATOR_ORDER) + [m for m in RESOURCE_MANAGER_ORDER if m not in ESTIMATOR_ORDER]
    return tuple(merged)


def order_method_names(names: Sequence[str], *, stage_hint: str | None = None) -> list[str]:
    """Return method names in stable thesis legend order."""
    canonical = _canonical_order(stage_hint)
    rank = {name: index for index, name in enumerate(canonical)}
    return sorted(names, key=lambda name: (rank.get(name, len(canonical)), name))


def order_methods_dict(
    methods: Mapping[str, Any],
    *,
    stage_hint: str | None = None,
) -> dict[str, Any]:
    ordered_names = order_method_names(list(methods.keys()), stage_hint=stage_hint)
    return {name: methods[name] for name in ordered_names}


def method_style_index(name: str, *, stage_hint: str | None = None) -> int:
    canonical = _canonical_order(stage_hint)
    if name in canonical:
        return canonical.index(name)
    return len(canonical) + hash(name) % 8


# Display names used in thesis figures; they match the prose and tables of the
# thesis rather than the code identifiers.
METHOD_LABELS: dict[str, str] = {
    "ls": "LS",
    "dft": "DFT",
    "lmmse": "LMMSE",
    "adaptive": "Adaptive",
    "neural": "Neural",
    "pso": "PSO",
    "ista": "ISTA",
    "static": "Static",
    "round_robin": "Round-robin",
    "max_throughput": "Max-throughput",
    "pf": "PF",
    "wmmse": "WMMSE",
    "queue_aware": "Queue-aware",
    "drl": "DRL",
    "reliability_drl": "Reliability-aware DRL",
    "cnn": "CNN",
}


def method_label(name: str) -> str:
    """Return the thesis display name for a method identifier."""
    return METHOD_LABELS.get(name, name)


# Results plots follow the diagram colour code by family: channel estimators
# (PHY) in blues, classical schedulers (MAC) in oranges and browns, learned
# methods in purple. Within a family the shades differ mainly in lightness, so
# they stay apart under colour-vision deficiency, and every method keeps its own
# marker. Red-oranges are avoided so no curve reads as the vermillion coupling.
METHOD_COLORS: dict[str, str] = {
    # PHY: classical and exploratory estimators
    "adaptive": "#06294a",
    "ls": "#0b5e9c",
    "dft": "#3a87c8",
    "lmmse": "#7db6e3",
    "pso": "#2c6f8f",
    "ista": "#5b9fbf",
    # MAC: classical schedulers
    "static": "#4a2e00",
    "round_robin": "#7a4c00",
    "max_throughput": "#a66900",
    "pf": "#d18b00",
    "wmmse": "#e6b23c",
    "queue_aware": "#b8975a",
    # Learned methods
    "neural": "#a2508a",
    "drl": "#a2508a",
    "reliability_drl": "#5b2366",
    "cnn": "#c58ab5",
}


def method_color(name: str, *, stage_hint: str | None = None) -> str:
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    index = method_style_index(name, stage_hint=stage_hint)
    return PALETTE[index % len(PALETTE)]


def method_marker(name: str, *, stage_hint: str | None = None) -> str:
    index = method_style_index(name, stage_hint=stage_hint)
    return MARKERS[index % len(MARKERS)]


def apply_thesis_rcparams(plt: Any) -> None:
    """Apply table-anchored thesis typography (Times New Roman) and consistent line weights."""
    from cycler import cycler

    plt.rcParams.update(
        {
            "font.family": "serif",
            # STIX ships with matplotlib and matches Times; the container has no Times.
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": TABLE_FONT_PT,
            "text.color": INK,
            "axes.labelsize": FIG_TITLE_PT,
            "axes.titlesize": FIG_TITLE_PT,
            "axes.titleweight": "normal",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=list(PALETTE)),
            "legend.fontsize": TABLE_FONT_PT,
            # Borderless white backing so an in-axes legend never sits on a curve.
            "legend.frameon": True,
            "legend.edgecolor": "none",
            "legend.facecolor": "white",
            "legend.framealpha": 0.9,
            "legend.fancybox": False,
            "xtick.labelsize": TABLE_FONT_PT,
            "ytick.labelsize": TABLE_FONT_PT,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
            "lines.markeredgewidth": 1.0,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "grid.linestyle": "-",
            "image.cmap": "Blues",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": THESIS_DPI,
            "savefig.dpi": THESIS_DPI,
            "savefig.bbox": "tight",
        }
    )


def style_ebno_axis(ax: Any, *, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(EBNO_XLABEL)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major")
