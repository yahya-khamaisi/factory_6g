#!/usr/bin/env python3
"""Generate thesis architecture figures as draw.io XML (semantic colour theme).

Colour carries meaning, the same meaning in every figure (Okabe-Ito based, so it
stays distinguishable for colour-vision deficiency):
  env      stone   environment and inputs (factory, channel, shared context)
  phy      blue    physical layer: estimation, equalisation, decoding
  mac      orange  medium access: scheduling and resource directives
  output   green   metrics and outputs
  coupling vermillion  the PHY->MAC coupling: estimate and error reach the scheduler
  process  grey    procedure steps; ``unused`` grey dashed = not used in Chapter 4
  learned  purple outline on top of the layer colour (neural estimator, DRL)
Each figure gets a small legend listing only the roles it uses. Figures carry no
in-image title, subtitle or boxed notes: the LaTeX caption is the title.

Connector routing: orthogonal edges with explicit waypoints in open corridors so
lines never pass over box fills.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from xml.etree.ElementTree import Element, SubElement, tostring

FIGDIR = Path(__file__).resolve().parents[2] / "thesis" / "figures" / "src"
SUNWAY_NAVY = "#233369"

# Canonical figure style (thesis/notes/CONTEXT.md → Table typography anchor, Figure typographic tiers)
FONT_FAMILY = "Times New Roman"
# XML pts are calibrated for printed legibility at \linewidth (~145 mm), not raw documentclass pts.
# Aug 2026: larger box text per supervisor feedback; wrap earlier so labels stay inside shapes.
FONT_TITLE = 24
FONT_SUBTITLE = 21
FONT_BOX = 22
# Sep 2026: notes and edge labels raised from 18 pt; at \linewidth an 18 pt label
# on a 1100-unit canvas prints at roughly 6.7 pt, below comfortable reading size.
FONT_NOTE = 20
FONT_EDGE = 20
FONT_BADGE = 22
# Times New Roman averages well under the 0.58 em/character that draw.io's own
# autosize assumes; 0.50 keeps estimated widths close to the rendered text.
CHAR_WIDTH_FACTOR = 0.50

# Slightly narrower landscape → larger printed text at LaTeX \linewidth.
CANVAS_LANDSCAPE = (1100, 980)
CANVAS_PORTRAIT = (900, 1250)

# Per-line character caps (Figure label line breaking). See thesis/notes/CONTEXT.md.
# Tighter caps force earlier wraps so larger fonts stay inside boxes.
LABEL_MAX_CHARS_BOX = 16
LABEL_MAX_CHARS_NARROW = 12
LABEL_MAX_CHARS_NOTE = 30
LABEL_MAX_CHARS_SUBTITLE = 46

_SEMANTIC_BREAKS = (" · ", " → ", "; ", " — ", " – ", ", ")

INK = "#1f2328"
MUTED = "#5f6368"
NAVY_TINT = "#eef0f6"
STROKE_W = 1.5
STROKE_W_KEY = 2.5

# Semantic role -> (fill, stroke). Strokes are darkened Okabe-Ito hues so labels
# and outlines keep contrast on the light fills.
SEMANTIC = {
    "env": ("#f1eee9", "#7a6c5d"),
    "phy": ("#e2eef8", "#0b5e9c"),
    "mac": ("#fdefd3", "#a66900"),
    "output": ("#dcf1e9", "#007a5a"),
    "coupling": ("#fbe4d8", "#c0510a"),
    "process": ("#f5f5f5", "#5f6368"),
    "unused": ("#ffffff", "#8a8a8a"),
    "training": ("#f5f5f5", "#5f6368"),
}
LEARNED_STROKE = "#a2508a"
LEGEND_LABELS = {
    "env": "Environment and inputs",
    "phy": "Physical layer (PHY)",
    "mac": "Scheduling (MAC)",
    "learned": "Learned component",
    "output": "Metrics and outputs",
    "coupling": "PHY\u2013MAC coupling",
    "process": "Procedure step",
    "unused": "Not used in Chapter 4",
    "training": "Used in training only",
}
LEGEND_ORDER = ("env", "phy", "mac", "learned", "coupling", "output", "process", "training", "unused")
# Pre-semantic role names still accepted, mapped to the closest meaning.
_LEGACY_ROLE = {
    "primary": "process", "secondary": "process", "warning": "process",
    "accent": "process", "neutral": "process", "success": "output", "key": "coupling",
}
PALETTE = {**SEMANTIC, **{k: SEMANTIC[v] for k, v in _LEGACY_ROLE.items()}}


def _semantic(role: str) -> str:
    return _LEGACY_ROLE.get(role, role)

EDGE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
    f"html=1;endArrow=block;endFill=1;endSize=7;strokeWidth={STROKE_W};strokeColor={INK};"
)
EDGE_DASHED = EDGE + "dashed=1;"


def _cell_value(text: str) -> str:
    """Return a draw.io label value with hard line breaks.

    Styles here set ``html=1``, so draw.io parses the label as HTML: a literal
    newline collapses to a space and only ``<br>`` forces a break.  Each line is
    escaped on its own and the breaks are re-inserted afterwards, so text in the
    line never turns into markup.  ElementTree escapes the ``<br>`` again when it
    serialises the attribute, which is what draw.io expects to read back.
    """
    normalized = text.replace("<br>", "\n")
    return "<br>".join(html.escape(line) for line in normalized.split("\n"))


def _wrap_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    """Split one paragraph into lines ≤ max_chars, preferring semantic boundaries."""
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]

    for sep in _SEMANTIC_BREAKS:
        if sep not in paragraph:
            continue
        parts = paragraph.split(sep)
        lines: list[str] = []
        current = parts[0].strip()
        sep_stripped = sep.strip()
        # Keep the separator visible when a wrap falls on it: trailing "," / ";"
        # attach tight; list bullets and dashes keep a leading space before the mark.
        if sep_stripped in {",", ";"}:
            trailing = sep_stripped
        elif sep_stripped in {"·", "—", "–", "→"}:
            trailing = f" {sep_stripped}"
        else:
            trailing = ""
        for part in parts[1:]:
            candidate = f"{current}{sep}{part}" if current else part.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(f"{current}{trailing}")
                current = part.strip()
        if current:
            lines.append(current)
        if len(lines) > 1:
            return lines

    words = paragraph.split()
    lines: list[str] = []
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


def normalize_label(text: str, *, max_chars: int = LABEL_MAX_CHARS_BOX) -> str:
    """Ensure multi-line labels; never leave long single-line cram (Figure label line breaking)."""
    if not text.strip():
        return text
    lines_out: list[str] = []
    for paragraph in text.replace("<br>", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines_out.extend(_wrap_paragraph(paragraph, max_chars=max_chars))
    return "<br>".join(lines_out)


def estimate_box_dims(
    label: str,
    *,
    font_pt: int = FONT_BOX,
    min_w: float = 72,
    min_h: float = 36,
    h_pad: float = 12,
    w_pad: float = 14,
    char_width_factor: float = CHAR_WIDTH_FACTOR,
    line_height_factor: float = 1.32,
) -> tuple[float, float]:
    """Minimum shape size so wrapped text fits at locked tier (Figure shape text fitting)."""
    lines = label.replace("<br>", "\n").split("\n")
    max_len = max((len(line) for line in lines), default=1)
    n_lines = max(len(lines), 1)
    w = max(min_w, max_len * font_pt * char_width_factor + 2 * w_pad)
    h = max(min_h, n_lines * font_pt * line_height_factor + 2 * h_pad)
    return w, h


class DrawioBuilder:
    def __init__(self, page_w: int = CANVAS_LANDSCAPE[0], page_h: int = CANVAS_LANDSCAPE[1]) -> None:
        self.page_w = page_w
        self.page_h = page_h
        self._id = 2
        self.cells: list[tuple[str, dict[str, Any]]] = []
        self._geom: dict[str, tuple[float, float, float, float]] = {}
        self.footnotes: list[str] = []
        self.roles_used: set[str] = set()
        self.legend = True

    def _nid(self) -> str:
        self._id += 1
        return str(self._id)

    def title(self, x: float, y: float, w: float, text: str) -> str:
        """No-op: the LaTeX caption is the figure title. Kept so builders read as before."""
        return self._nid()

    def subtitle(self, x: float, y: float, w: float, text: str) -> str:
        """No-op: context belongs in the caption, not inside the image."""
        return self._nid()

    def heading(self, x: float, y: float, w: float, text: str) -> str:
        """Bold left-aligned label naming a panel, lane or group inside the figure."""
        tid = self._nid()
        fit_chars = max(24, int(w / (FONT_SUBTITLE * CHAR_WIDTH_FACTOR)))
        text = normalize_label(text, max_chars=max(LABEL_MAX_CHARS_SUBTITLE, fit_chars))
        n_lines = max(1, len(text.split("<br>")))
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_SUBTITLE};fontStyle=1;fontColor={INK};"
        )
        h = n_lines * int(FONT_SUBTITLE * 1.4) + 8
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return tid

    def note(self, x: float, y: float, w: float, h: float, text: str, role: str = "neutral") -> str:
        tid = self._nid()
        fit_chars = max(24, int((w - 28) / (FONT_NOTE * CHAR_WIDTH_FACTOR)))
        text = normalize_label(text, max_chars=max(LABEL_MAX_CHARS_NOTE, fit_chars))
        # Plain italic annotation; no frame or fill (role kept for call compatibility).
        style = (
            f"text;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;fontColor={MUTED};"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_NOTE};fontStyle=2;align=center;verticalAlign=middle;"
        )
        auto_w, auto_h = estimate_box_dims(text, font_pt=FONT_NOTE, min_w=w, min_h=h)
        w, h = max(w, auto_w), max(h, auto_h)
        self._geom[tid] = (x, y, w, h)
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return tid

    def footnote(self, x: float, y: float, w: float, h: float, text: str, role: str = "neutral") -> str:
        """Figure-level remark: not drawn. Its content belongs in the LaTeX caption.

        Geometry is still recorded so builders that stack on it keep working.
        """
        tid = self._nid()
        self._geom[tid] = (x, y, w, 0)
        self.footnotes.append(text)
        return tid

    def step_badge(self, x: float, y: float, n: int) -> str:
        cid = self._nid()
        badge = max(34, int(FONT_BADGE * 2.1))
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeWidth={STROKE_W};"
            f"strokeColor={INK};fontColor={INK};fontFamily={FONT_FAMILY};fontSize={FONT_BADGE};fontStyle=1;"
            f"align=center;verticalAlign=middle;"
        )
        self.cells.append(
            (cid, {"value": str(n), "style": style, "vertex": "1", "step_badge": True, "x": x, "y": y, "w": badge, "h": badge})
        )
        return cid

    def step_badge_left(self, node_id: str, n: int, gap: float = 10) -> str:
        """Numbered circle centred on the node's left edge—clear of titles above."""
        x, y, _w, h = self._geom[node_id]
        badge = max(34, int(FONT_BADGE * 2.1))
        return self.step_badge(x - badge - gap, y + (h - badge) / 2, n)

    def step_badge_near_tl(self, node_id: str, n: int, gap: float = 8) -> str:
        """Numbered circle just outside top-left corner—close, no overlap with node fill."""
        x, y, _w, _h = self._geom[node_id]
        badge = max(34, int(FONT_BADGE * 2.1))
        bx = x - badge - gap
        by = y - badge - gap
        return self.step_badge(bx, by, n)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        role: str = "primary",
        dashed: bool = False,
        ellipse: bool = False,
        *,
        fixed_width: bool = False,
        max_chars: int | None = None,
        learned: bool = False,
    ) -> str:
        cid = self._nid()
        role = _semantic(role)
        self.roles_used.add(role)
        if learned:
            self.roles_used.add("learned")
        if role in ("unused", "training"):
            dashed = True
        explicit_max_chars = max_chars is not None
        if max_chars is None:
            max_chars = LABEL_MAX_CHARS_NARROW if fixed_width else LABEL_MAX_CHARS_BOX
        if fixed_width:
            # Re-wrap to the characters that actually fit the locked column width,
            # otherwise long wrapped lines spill outside the shape fill. When the
            # caller did not pick a cap, the width alone decides it.
            fit_chars = int((w - 2 * 14) / (FONT_BOX * CHAR_WIDTH_FACTOR))
            max_chars = fit_chars if not explicit_max_chars else min(max_chars, fit_chars)
            max_chars = max(6, max_chars)
        label = normalize_label(label, max_chars=max_chars)
        fill, stroke = PALETTE[role]
        auto_w, auto_h = estimate_box_dims(label, font_pt=FONT_BOX, min_w=w, min_h=h)
        if fixed_width:
            h = max(h, auto_h)
        else:
            w, h = max(w, auto_w), max(h, auto_h)
        parts = [
            "whiteSpace=wrap",
            "html=1",
            f"fillColor={fill}",
            f"strokeColor={LEARNED_STROKE if learned else stroke}",
            f"strokeWidth={2.6 if learned else (STROKE_W_KEY if role == 'coupling' else STROKE_W)}",
            f"fontColor={INK}",
            f"fontFamily={FONT_FAMILY}",
            f"fontSize={FONT_BOX}",
            "align=center",
            "verticalAlign=middle",
            "rounded=0",
        ]
        if dashed:
            parts.append("dashed=1")
        if ellipse:
            parts.insert(0, "ellipse")
        style = ";".join(parts)
        self._geom[cid] = (x, y, w, h)
        self.cells.append((cid, {"value": _cell_value(label), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return cid

    def center_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        color: str = MUTED,
        font_pt: int = FONT_NOTE,
    ) -> str:
        tid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW)
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            f"fontFamily={FONT_FAMILY};fontSize={font_pt};fontColor={color};"
        )
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return tid

    def group_rect(self, x: float, y: float, w: float, h: float, role: str | None = None) -> str:
        """Frame around a group; with a role, the frame takes that role's stroke colour."""
        gid = self._nid()
        stroke = SEMANTIC[_semantic(role)][1] if role else "#a8a8a8"
        style = f"rounded=0;strokeColor={stroke};strokeWidth=1.2;fillColor=none;"
        self._geom[gid] = (x, y, w, h)
        self.cells.append((gid, {"value": "", "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return gid

    def _center(self, cid: str) -> tuple[float, float]:
        x, y, w, h = self._geom[cid]
        return x + w / 2, y + h / 2

    def move_y(self, node_id: str, y: float) -> None:
        x, _y, w, h = self._geom[node_id]
        self._geom[node_id] = (x, y, w, h)
        for cid, data in self.cells:
            if cid == node_id:
                data["y"] = y

    def bottom(self, node_id: str) -> float:
        x, y, w, h = self._geom[node_id]
        return y + h

    def equalize_heights(self, node_ids: list[str]) -> float:
        """Force a row of boxes to a shared height (tallest wins) for grid alignment."""
        if not node_ids:
            return 0.0
        max_h = max(self._geom[nid][3] for nid in node_ids)
        id_set = set(node_ids)
        for cid, data in self.cells:
            if cid in id_set and "h" in data:
                data["h"] = max_h
                x, y, w, _ = self._geom[cid]
                self._geom[cid] = (x, y, w, max_h)
        return max_h

    def row_label(self, x: float, y: float, w: float, h: float, text: str) -> str:
        """Solid Sunway-navy row label (research-framework left column)."""
        cid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW)
        style = (
            f"whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#5f6368;strokeWidth={STROKE_W};"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_BOX};fontColor={INK};fontStyle=1;"
            f"align=center;verticalAlign=middle;rounded=0;"
        )
        auto_w, auto_h = estimate_box_dims(text, font_pt=FONT_BOX, min_w=w, min_h=h)
        h = max(h, auto_h)
        # Keep column width fixed; wrap label text instead of widening into content area.
        self._geom[cid] = (x, y, w, h)
        self.cells.append((cid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return cid

    def edge(
        self,
        src: str,
        dst: str,
        dashed: bool = False,
        color: str | None = None,
        points: list[tuple[float, float]] | None = None,
        label: str = "",
        thick: bool = False,
        extra_style: str = "",
    ) -> None:
        eid = self._nid()
        style = EDGE_DASHED if dashed else EDGE
        # Edge colour follows meaning: a thick edge is the PHY-MAC coupling; a
        # semantic role name colours the edge with that role's stroke; anything
        # else stays ink.
        edge_color = INK
        if thick:
            edge_color = SEMANTIC["coupling"][1]
            self.roles_used.add("coupling")
        elif color in SEMANTIC:
            edge_color = SEMANTIC[color][1]
        style = style.replace(f"strokeColor={INK}", f"strokeColor={edge_color}")
        if thick:
            style = style.replace(f"strokeWidth={STROKE_W}", f"strokeWidth={STROKE_W_KEY}")
        if label:
            label_color = edge_color
            style += (
                f"fontFamily={FONT_FAMILY};fontSize={FONT_EDGE};fontColor={label_color};"
                "labelBackgroundColor=#ffffff;fontStyle=2;"
            )
        style += extra_style
        self.cells.append(
            (
                eid,
                {
                    "value": _cell_value(normalize_label(label, max_chars=LABEL_MAX_CHARS_NOTE)) if label else "",
                    "style": style,
                    "edge": "1",
                    "source": src,
                    "target": dst,
                    "points": points or [],
                },
            )
        )

    def route(
        self,
        src: str,
        dst: str,
        waypoints: list[tuple[float, float]],
        dashed: bool = False,
        color: str | None = None,
        label: str = "",
        thick: bool = False,
    ) -> None:
        self.edge(src, dst, dashed=dashed, color=color, points=waypoints, label=label, thick=thick)

    def add_legend(self) -> None:
        """Row of swatches below the drawing for the roles this figure uses."""
        roles = [r for r in LEGEND_ORDER if r in self.roles_used]
        if len(roles) < 2 or not self._geom:
            return
        left = min(g[0] for g in self._geom.values())
        right = max(g[0] + g[2] for g in self._geom.values())
        # Clear the lowest box and any connector routed just beneath it.
        lowest_edge = max((py for _cid, c in self.cells if c.get("edge")
                           for _px, py in (c.get("points") or [])), default=0.0)
        y = max(max(g[1] + g[3] for g in self._geom.values()), lowest_edge) + 40
        sw, sh, gap = 30, 20, 14
        widths = [sw + 8 + len(LEGEND_LABELS[r]) * FONT_NOTE * CHAR_WIDTH_FACTOR + 8 for r in roles]
        rows: list[list[int]] = [[]]
        cur = 0.0
        for i, w in enumerate(widths):
            if cur + w > (right - left) and rows[-1]:
                rows.append([])
                cur = 0.0
            rows[-1].append(i)
            cur += w + gap
        for r_i, row in enumerate(rows):
            rw = sum(widths[i] for i in row) + gap * (len(row) - 1)
            x = left + max(0.0, ((right - left) - rw) / 2)
            yy = y + r_i * (sh + 14)
            for i in row:
                role = roles[i]
                if role == "learned":
                    fill, stroke, sw_w = "#ffffff", LEARNED_STROKE, 2.6
                else:
                    fill, stroke = SEMANTIC[role]
                    sw_w = STROKE_W_KEY if role == "coupling" else STROKE_W
                dash = "dashed=1;" if role in ("unused", "training") else ""
                sid = self._nid()
                self.cells.append((sid, {"value": " ", "style": (
                    f"rounded=0;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw_w};{dash}"),
                    "vertex": "1", "x": x, "y": yy, "w": sw, "h": sh}))
                tid = self._nid()
                self.cells.append((tid, {"value": _cell_value(LEGEND_LABELS[role]), "style": (
                    f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
                    f"fontFamily={FONT_FAMILY};fontSize={FONT_NOTE};fontColor={INK};"),
                    "vertex": "1", "x": x + sw + 8, "y": yy - 4, "w": widths[i] - sw - 8, "h": sh + 8}))
                x += widths[i] + gap

    def to_xml(self) -> str:
        if self.legend:
            self.add_legend()
        mxfile = Element("mxfile", {"host": "app.diagrams.net", "agent": "build_thesis_drawio", "version": "24.7.17"})
        diagram = SubElement(mxfile, "diagram", {"id": "page1", "name": "Page-1"})
        model = SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1200",
                "dy": "800",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(self.page_w),
                "pageHeight": str(self.page_h),
                "math": "0",
                "shadow": "0",
            },
        )
        root = SubElement(model, "root")
        SubElement(root, "mxCell", {"id": "0"})
        SubElement(root, "mxCell", {"id": "1", "parent": "0"})

        edges = [c for c in self.cells if c[1].get("edge")]
        verts = [c for c in self.cells if c[1].get("vertex")]
        # Z-order: connectors → grouping frames → labelled nodes → step badges (badges above connectors).
        bg_verts = [c for c in verts if not c[1].get("value")]
        fg_nodes = [c for c in verts if c[1].get("value") and not c[1].get("step_badge")]
        step_badges = [c for c in verts if c[1].get("step_badge")]

        for cid, data in edges + bg_verts + fg_nodes + step_badges:
            attrs: dict[str, str] = {"id": cid, "parent": "1", "style": data["style"]}
            if "value" in data:
                attrs["value"] = data["value"]
            if data.get("vertex"):
                attrs["vertex"] = "1"
            if data.get("edge"):
                attrs["edge"] = "1"
                attrs["source"] = data["source"]
                attrs["target"] = data["target"]
            cell = SubElement(root, "mxCell", attrs)
            if "x" in data:
                SubElement(
                    cell,
                    "mxGeometry",
                    {"x": str(data["x"]), "y": str(data["y"]), "width": str(data["w"]), "height": str(data["h"]), "as": "geometry"},
                )
            elif data.get("edge"):
                geo = SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
                pts = data.get("points") or []
                if pts:
                    arr = SubElement(geo, "Array", {"as": "points"})
                    for px, py in pts:
                        SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})

        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + tostring(mxfile, encoding="unicode")


def build_integrated_pipeline() -> str:
    """Two-lane flow: a PHY link-level chain on top feeds a MAC stage below.

    The single vertical arrow between the lanes is the point of the figure --
    the scheduler is driven by the estimator's *estimated* CSI (and its error),
    in a fixed order, on one shared set of channel samples, without joint
    end-to-end training.
    """
    pw, ph = 1180, 792
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    col_gap = 18
    sec = "mac"
    grn = "output"
    ntl = "env"

    d.title(mx, 12, cw, "How the two stages are evaluated together")

    # --- Shared-samples band feeding the chain -----------------------------
    band = d.box(
        mx, 62, cw, 58,
        "Same channel samples per batch  ·  Eb/N0 sweep  ·  identical across the methods compared",
        "env", fixed_width=True, max_chars=96,
    )

    # --- PHY lane --------------------------------------------------------
    phy_y, phy_h = 196, 118
    d.heading(mx, 150, cw, "PHY link-level chain (held fixed)")
    n = 5
    bw = (cw - (n - 1) * col_gap) // n
    xs = [mx + i * (bw + col_gap) for i in range(n)]
    cx = [x + bw / 2 for x in xs]
    tx = d.box(xs[0], phy_y, bw, phy_h, "OFDM<br>transmit", "phy", fixed_width=True, max_chars=14)
    ch = d.box(xs[1], phy_y, bw, phy_h, "Factory<br>channel", "env", fixed_width=True, max_chars=14)
    ce = d.box(xs[2], phy_y, bw, phy_h, "Channel<br>estimation", "phy", fixed_width=True, max_chars=14)
    dec = d.box(xs[3], phy_y, bw, phy_h, "Equalise +<br>decode", "phy", fixed_width=True, max_chars=14)
    ber = d.box(xs[4], phy_y, bw, phy_h, "Decoded<br>BER", "output", fixed_width=True, max_chars=14)
    for a, b in [(tx, ch), (ch, ce), (ce, dec), (dec, ber)]:
        d.edge(a, b)
    d.edge(band, ch, color=ntl, points=[(cx[1], 150)])

    # --- MAC lane -------------------------------------------------------
    mac_y, mac_h = 452, 118
    d.heading(mx, 406, cw, "MAC resource management (the stage compared)")
    rm = d.box(xs[2], mac_y, bw, mac_h, "Resource<br>manager", "mac", fixed_width=True, max_chars=14)
    sch = d.box(xs[3], mac_y, bw, mac_h, "Scheduling +<br>power", "mac", fixed_width=True, max_chars=14)
    d.edge(rm, sch, color=sec)

    # --- Coupling: the whole point of the figure ----------------------
    d.edge(
        ce, rm, color=sec, thick=True,
        label="estimated CSI + error variance (not perfect CSI)",
    )

    # --- Metrics: everything reported together ----------------------
    met_y = 608
    met = d.box(
        mx, met_y, cw, 56,
        "Reported together, on the same runs:  BER · throughput · latency · power · runtime",
        "output", fixed_width=True, max_chars=96,
    )
    d.edge(ber, met, color=grn, points=[(cx[4], met_y - 12)])
    d.edge(sch, met, color=grn, points=[(cx[3], met_y - 12)])

    # --- One-line reading of the figure -------------------------
    d.box(
        mx, met_y + 88, cw, 46,
        "Fixed order: estimate, then schedule  ·  shared samples  ·  not trained end-to-end",
        "process", fixed_width=True, max_chars=96,
    )
    return d.to_xml()


def build_lr_taxonomy() -> str:
    """Portrait canvas; vertical top-down tree with CE/RM pillar branches."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    col_gap = 24
    col_w = (cw - col_gap) // 2
    lx, rx = mx, mx + col_w + col_gap

    def bottom(node_id: str) -> float:
        x, y, w, h = d._geom[node_id]
        return y + h

    d.title(mx, 16, cw, "Smart-factory wireless literature map")
    d.subtitle(
        mx,
        52,
        cw,
        "Chapter 2 — classical vs ML families for CE and RM; decoupled vs integrated PHY–MAC evaluation.",
    )

    # Full-width banner boxes: allow long lines (draw.io wraps to box width anyway);
    # a small max_chars here forces many short lines and an over-tall figure.
    wide_mc = 58

    y = 108
    req = d.box(mx, y, cw, 52, "Industrial requirements<br>URLLC · latency · mobility · heterogeneous UTs", "accent", max_chars=wide_mc)
    y = bottom(req) + row_gap

    root = d.box(
        mx,
        y,
        cw,
        52,
        "6G / B5G smart-factory wireless<br>industrial IoT · AGV/robotics · factory RF · network slicing",
        "primary",
        max_chars=wide_mc,
    )
    y = bottom(root) + row_gap

    met = d.box(mx, y, cw, 52, "Evaluation metrics<br>BER · throughput · latency · power · fairness", "success", max_chars=wide_mc)
    y = bottom(met) + row_gap

    dec = d.box(
        mx,
        y,
        cw,
        52,
        "Typical literature: decoupled CE/RM benchmarks<br>CE curves without MAC impact · perfect-CSI schedulers",
        "neutral",
        dashed=True,
        max_chars=wide_mc,
    )
    y = bottom(dec) + row_gap + 10

    d.heading(lx, y, col_w, "Channel estimation branch")
    d.heading(rx, y, col_w, "Resource management branch")
    y += 36

    col_mc = 34

    ce_cls = d.box(
        lx,
        y,
        col_w,
        54,
        "Classical channel estimation<br>LS · DFT · LMMSE<br>pilot-aided OFDM interpolation",
        "primary",
        max_chars=col_mc,
        fixed_width=True,
    )
    rm_cls = d.box(
        rx,
        y,
        col_w,
        54,
        "Classical resource management<br>PF · WMMSE · max-throughput<br>static · round-robin · queue-aware",
        "primary",
        max_chars=col_mc,
        fixed_width=True,
    )
    y = max(bottom(ce_cls), bottom(rm_cls)) + row_gap

    ce_ml = d.box(
        lx,
        y,
        col_w,
        54,
        "Learned channel estimation<br>CNN denoising · deep unfolding<br>learned MMSE · adaptive hybrids",
        "secondary",
        max_chars=col_mc,
        fixed_width=True,
    )
    rm_ml = d.box(
        rx,
        y,
        col_w,
        54,
        "Learned resource management<br>DRL actors · learning-to-optimise<br>GNN schedulers",
        "secondary",
        max_chars=col_mc,
        fixed_width=True,
    )
    y = max(bottom(ce_ml), bottom(rm_ml)) + row_gap

    ce_ctx = d.box(lx, y, col_w, 48, "Factory channel context<br>Rayleigh · Rician · TR 38.901 · ray-traced", "accent", max_chars=col_mc, fixed_width=True)
    rm_ctx = d.box(
        rx,
        y,
        col_w,
        48,
        "MAC objectives<br>scheduling · power control<br>URLLC QoS · multi-UT fairness",
        "warning",
        max_chars=col_mc,
        fixed_width=True,
    )
    y = max(bottom(ce_ctx), bottom(rm_ctx)) + row_gap + 16

    gap = d.box(
        mx,
        y,
        cw,
        60,
        "RESEARCH GAP (this thesis)<br>Integrated CE + RM evaluation on shared channel realisations<br>"
        "end-to-end BER under factory geometry · staged coupled benchmarking (P1–P3)",
        "secondary",
        max_chars=wide_mc,
    )
    y = bottom(gap) + row_gap + 12

    leg_h = 34
    leg_w = 118
    leg_gap = 14
    d.box(mx, y, leg_w, leg_h, "Classical", "primary")
    d.box(mx + leg_w + leg_gap, y, leg_w, leg_h, "ML / DRL", "secondary")
    d.box(mx + 2 * (leg_w + leg_gap), y, leg_w + 24, leg_h, "Factory context", "accent")

    d.edge(req, root)
    d.edge(root, met)
    d.edge(met, dec)

    branch_y = d._geom[ce_cls][1] - 12
    dec_cx, _ = d._center(dec)
    ce_cx, _ = d._center(ce_cls)
    rm_cx, _ = d._center(rm_cls)
    d.route(dec, ce_cls, [(dec_cx, branch_y), (ce_cx, branch_y)])
    d.route(dec, rm_cls, [(dec_cx, branch_y), (rm_cx, branch_y)])

    for a, b in [(ce_cls, ce_ml), (ce_ml, ce_ctx), (rm_cls, rm_ml), (rm_ml, rm_ctx)]:
        d.edge(a, b)

    merge_y = d._geom[gap][1] - 14
    gap_cx, _ = d._center(gap)
    ctx_ce_x, _ = d._center(ce_ctx)
    ctx_rm_x, _ = d._center(rm_ctx)
    d.route(ce_ctx, gap, [(ctx_ce_x, merge_y), (gap_cx, merge_y)])
    d.route(rm_ctx, gap, [(ctx_rm_x, merge_y), (gap_cx, merge_y)])

    return d.to_xml()


def build_phy_stack() -> str:
    """Portrait canvas; vertical PHY chain top-to-bottom, matching Model.run_batch."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    stack_w = 520
    bx = mx + (cw - stack_w) // 2

    d.title(mx, 16, cw, "Physical-layer stack")
    d.subtitle(
        mx,
        52,
        cw,
        "Coded OFDM uplink evaluated once per Monte Carlo batch.",
    )

    transmit: list[tuple[str, str]] = [
        ("Information bits · 1536 per user terminal", "env"),
        ("5G NR LDPC encoder · rate 1/2 · 3072 coded bits", "phy"),
        ("QPSK mapper · 2 bits per symbol", "phy"),
        (
            "OFDM resource grid · 14 symbols × 128 subcarriers · "
            "Kronecker pilots on symbols 2, 11",
            "phy",
        ),
        (
            "Factory channel · Rayleigh, Rician, TR 38.901 UMi or ray-traced hall · plus white Gaussian noise",
            "env",
        ),
    ]
    receive: list[tuple[str, str]] = [
        ("Channel estimator · LS, DFT, LMMSE, adaptive or neural", "phy"),
        ("LMMSE equaliser · uses the estimate and its error variance", "phy"),
        ("Soft demapper · log-likelihood ratios", "phy"),
        ("5G NR LDPC decoder · belief propagation · 20 iterations", "phy"),
        ("Bit and block error counting · per batch", "output"),
    ]

    y = 108
    ids: list[str] = []
    for label, role in transmit:
        ids.append(d.box(bx, y, stack_w, 56, label.replace(" · ", "<br>"), role, max_chars=34))
        y = d.bottom(ids[-1]) + row_gap

    subtitle_y = y + 14
    d.heading(bx, subtitle_y, stack_w, "Receiver chain: estimate, equalise, decode, count")
    y = subtitle_y + 44
    rx_top = y

    for label, role in receive:
        ids.append(d.box(bx, y, stack_w, 56, label.replace(" · ", "<br>"), role, max_chars=34))
        y = d.bottom(ids[-1]) + row_gap

    d.group_rect(bx - 12, rx_top - 10, stack_w + 24, d.bottom(ids[-1]) - rx_top + 18, role="phy")

    grid = ids[3]
    _gx, grid_y, _gw, grid_h = d._geom[grid]
    rm_w = cw - (bx + stack_w + 20) - mx
    rm = d.box(
        bx + stack_w + 48,
        grid_y,
        max(rm_w, 210),
        max(grid_h, 96),
        "MAC resource directives<br>active-UT mask and per-UT power<br>from the scheduler stage",
        "mac",
        dashed=True,
        fixed_width=True,
    )

    for a, b in zip(ids[:-1], ids[1:]):
        d.edge(a, b)
    # Centre the directive box on the grid row so the dashed arrow runs straight.
    _gx, g_y, _gw, g_h = d._geom[grid]
    d.move_y(rm, g_y + g_h / 2 - d._geom[rm][3] / 2)
    r_cy = d._center(rm)[1]
    entry_y = min(max((r_cy - g_y) / g_h, 0.2), 0.8)
    d.edge(
        rm, grid, dashed=True, color="mac",
        extra_style=f"exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY={entry_y:.3f};entryDx=0;entryDy=0;",
    )

    d.footnote(
        mx,
        d.bottom(ids[-1]) + row_gap + 10,
        cw,
        48,
        "Within a batch every estimator sees the same channel and noise realisation (P1).",
        "neutral",
    )
    return d.to_xml()


def build_monte_carlo() -> str:
    """Landscape canvas; the two stage loops side by side, run left then right (P3)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 20
    cw = pw - 2 * mx
    row_gap = 14
    col_gap = 70
    gw = (cw - col_gap) // 2
    badge = max(34, int(FONT_BADGE * 2.1))
    pad_l = badge + 26
    bw = gw - pad_l - 16

    y = 20
    start = d.box(
        mx + pad_l, y, bw, 64,
        "Load configuration; seed Python, NumPy, TensorFlow and Sionna",
        "process", ellipse=True, max_chars=34,
    )
    d.step_badge_left(start, 1)
    top = d.bottom(start) + 40

    def stage(gx: float, labels: list[tuple[str, str]]) -> list[str]:
        ids = [d.box(gx + pad_l, top, bw, 64, text, role, fixed_width=True, max_chars=26) for text, role in labels]
        for src, dst in zip(ids, ids[1:]):
            d.edge(src, dst)
        return ids

    lx = mx
    rx = mx + gw + col_gap
    s1 = stage(lx, [
        ("For each batch and each open Eb/N0 point", "process"),
        ("Draw one shared context: channel, noise, bits (P1)", "env"),
        ("Run every estimator with neutral MAC masks", "phy"),
        ("Accumulate metrics; close the point when the stopping rule fires", "output"),
    ])
    s2 = stage(rx, [
        ("For each batch and each open Eb/N0 point", "process"),
        ("Draw one shared context; probe adaptive-estimator feedback (P2)", "coupling"),
        ("Run every scheduler, apply its directives, then decode", "mac"),
        ("Accumulate metrics; close the point when the stopping rule fires", "output"),
    ])
    # Rows line up across the two columns: shared height, then stack.
    yy = top + 52
    for a_id, b_id in zip(s1, s2):
        h = d.equalize_heights([a_id, b_id])
        d.move_y(a_id, yy)
        d.move_y(b_id, yy)
        yy += h + row_gap
    for i, nid in enumerate(s1 + s2):
        d.step_badge_left(nid, i + 2)
    bottom = yy - row_gap + 18
    g1 = d.group_rect(lx, top, gw, bottom - top, role="phy")
    g2 = d.group_rect(rx, top, gw, bottom - top, role="mac")
    d.heading(lx + 14, top + 10, gw - 28, "Estimator stage (RQ1)")
    d.heading(rx + 14, top + 10, gw - 28, "Resource-manager stage (RQ2)")

    # Straight drop from the start node into the stage frame below it.
    s_cx, _ = d._center(start)
    d.edge(start, g1, extra_style=f"entryX={(s_cx - lx) / gw:.4f};entryY=0;entryDx=0;entryDy=0;")
    d.edge(g1, g2)

    y = bottom + 40
    classify = d.box(
        rx + pad_l, y, bw, 64,
        "Classify each point as resolved (at least 30 bit errors) or upper bound",
        "output", fixed_width=True, max_chars=30,
    )
    d.step_badge_left(classify, 10)
    c_cx, _ = d._center(classify)
    end = d.box(
        lx + pad_l, y, bw, d._geom[classify][3],
        "Write stage results, run summary and plots",
        "output", ellipse=True, fixed_width=True, max_chars=26,
    )
    d.edge(g2, classify, extra_style=f"exitX={(c_cx - rx) / gw:.4f};exitY=1;exitDx=0;exitDy=0;")
    # Leave from the badge-free bottom edge so the arrow never crosses badge 10.
    d.route(classify, end, [(c_cx, d.bottom(classify) + 24), (d._center(end)[0], d.bottom(classify) + 24)])
    d.page_h = int(d.bottom(classify) + 60)
    return d.to_xml()


def build_factory_topology() -> str:
    """Landscape canvas; deployment profile and how it parameterises the channel models."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 16
    col_gap = 26

    d.title(mx, 16, cw, "Small-factory deployment profile")
    d.subtitle(
        mx,
        52,
        cw,
        "Deployment, radio configuration and the channel models they drive.",
    )

    col_w = (cw - 2 * col_gap) // 3
    head_y = 100
    top = head_y + 34

    # --- Column 1: hall and terminals ---
    d.heading(mx, head_y, col_w, "Deployment")
    hall = d.box(
        mx, top, col_w, 60,
        "Factory hall<br>15 m × 15 m × 5 m<br>five metal machines",
        "env", fixed_width=True,
    )
    bs = d.box(
        mx, d.bottom(hall) + row_gap, col_w, 60,
        "Base station<br>eight antenna ports<br>four dual-polarised elements",
        "env", fixed_width=True,
    )
    ut = d.box(
        mx, d.bottom(bs) + row_gap, col_w, 60,
        "Four user terminals<br>one antenna each<br>stationary in the locked runs",
        "env", fixed_width=True,
    )

    # --- Column 2: radio configuration ---
    cx2 = mx + col_w + col_gap
    d.heading(cx2, head_y, col_w, "Radio configuration")
    numer = d.box(
        cx2, top, col_w, 60,
        "Carrier 3.5 GHz<br>128-point FFT<br>30 kHz subcarrier spacing",
        "phy", fixed_width=True,
    )
    link = d.box(
        cx2, d.bottom(numer) + row_gap, col_w, 60,
        "Uplink QPSK<br>LDPC rate 1/2<br>pilots on two OFDM symbols",
        "phy", fixed_width=True,
    )

    # --- Column 3: channel models ---
    cx3 = cx2 + col_w + col_gap
    d.heading(cx3, head_y, col_w, "Channel models")
    stoch = d.box(
        cx3, top, col_w, 60,
        "Rayleigh and Rician<br>block fading<br>Rician K-factor 1",
        "env", fixed_width=True,
    )
    tr = d.box(
        cx3, d.bottom(stoch) + row_gap, col_w, 60,
        "TR 38.901 UMi<br>small-scale fading only<br>path loss and shadowing off",
        "env", fixed_width=True,
    )
    d.box(
        cx3, d.bottom(tr) + row_gap, col_w, 60,
        "Ray-traced hall<br>Sionna RT in this layout<br>direct and reflected paths",
        "env", fixed_width=True,
    )
    return d.to_xml()


def build_estimator_comparison() -> str:
    """Landscape canvas; shared LS front end with four refinement branches."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 16

    d.title(mx, 16, cw, "Channel-estimation comparators")
    d.subtitle(
        mx,
        52,
        cw,
        "Every comparator reuses the same least-squares front end; only the refinement differs.",
    )

    in_w = 210
    branch_w = 320
    out_w = 230
    gap = 30
    bx_in = mx
    bx_branch = bx_in + in_w + gap
    bx_out = bx_branch + branch_w + gap

    top = 120
    branches = [
        ("Least squares<br>pilot inversion only", "phy", False),
        ("DFT denoising<br>keep the first 20 delay taps", "phy", False),
        ("LMMSE smoothing<br>exponential correlation, ρ = 0.98", "phy", False),
        ("Adaptive hybrid<br>branch chosen by a signal-quality proxy", "phy", False),
        ("Neural residual<br>convolutional correction added to LS", "phy", True),
    ]
    y = top
    branch_ids = []
    for label, role, learned in branches:
        branch_ids.append(d.box(bx_branch, y, branch_w, 64, label, role, fixed_width=True,
                                learned=learned))
        y = d.bottom(branch_ids[-1]) + row_gap
    span = d.bottom(branch_ids[-1]) - top
    mid_y = top + span / 2

    front = d.box(
        bx_in, mid_y - 100, in_w, 72,
        "Pilot observations<br>and noise variance",
        "env", fixed_width=True,
    )
    ls = d.box(
        bx_in, d.bottom(front) + row_gap, in_w, 72,
        "LS estimate<br>nearest-neighbour<br>interpolation",
        "phy", fixed_width=True,
    )
    out = d.box(
        bx_out, mid_y - 46, out_w, 92,
        "Channel estimate<br>and error variance<br>to the LMMSE equaliser",
        "phy", fixed_width=True,
    )

    d.edge(front, ls)
    fan_x = bx_in + in_w + gap // 2
    ls_cy = d._center(ls)[1]
    for bid in branch_ids:
        _bx, bcy = d._center(bid)
        d.route(ls, bid, [(fan_x, ls_cy), (fan_x, bcy)])
    merge_x = bx_branch + branch_w + gap // 2
    out_cy = d._center(out)[1]
    for bid in branch_ids:
        _bx, bcy = d._center(bid)
        d.route(bid, out, [(merge_x, bcy), (merge_x, out_cy)])

    ny = max(d.bottom(branch_ids[-1]), d.bottom(out), d.bottom(ls)) + row_gap + 12
    n1 = d.footnote(
        mx, ny, cw, 48,
        "Only the neural branch needs offline training; the other four are closed-form or rule-based.",
        "neutral",
    )
    d.footnote(
        mx, d.bottom(n1) + 12, cw, 48,
        "Within a batch every estimator sees the same channel and noise (P1). The shared "
        "comparison in Chapter 4 is LS against neural; the adaptive estimator serves every scheduler stage.",
        "accent",
    )
    d.page_h = max(ph, int(d.bottom(n1) + 140))
    return d.to_xml()


def build_rm_comparison() -> str:
    """Landscape canvas; schedulers grouped by the state each one consumes."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 18

    d.title(mx, 16, cw, "Resource-management comparators")
    d.subtitle(
        mx,
        52,
        cw,
        "Eight schedulers grouped by the state each one consumes; all emit the same directives.",
    )

    state_w = 250
    sched_w = 330
    out_w = 250
    gap = 30
    sx = mx
    cx = sx + state_w + gap
    ox = cx + sched_w + gap

    tiers = [
        (
            "No channel state:<br>configuration or cycle counter",
            "Static — all terminals active<br>Round-robin — two terminals in turn",
            "mac",
        ),
        (
            "Per-terminal channel energy<br>from the estimated channel",
            "Max-throughput — two strongest terminals<br>"
            "Proportional fair — rate over moving average",
            "mac",
        ),
        (
            "Channel energy plus extra state:<br>"
            "interference matrix or virtual backlogs",
            "WMMSE — iterative weighted MMSE power control<br>"
            "Queue-aware — drift-plus-penalty priority",
            "mac",
        ),
        (
            "Normalised policy state:<br>energy, fairness debt and Eb/N0",
            "DRL — PPO, fairness-weighted throughput reward<br>Reliability-aware DRL — PPO, log-BER reward minus power",
            "mac-learned",
        ),
    ]

    y = 120
    state_ids, sched_ids = [], []
    for state_label, sched_label, role in tiers:
        s = d.box(sx, y, state_w, 76, state_label, "env", fixed_width=True)
        c = d.box(cx, y, sched_w, 76, sched_label, role.split("-")[0], fixed_width=True,
                  learned=role.endswith("-learned"))
        h = d.equalize_heights([s, c])
        state_ids.append(s)
        sched_ids.append(c)
        d.edge(s, c)
        y += h + row_gap

    span_top = 120
    span_bot = d.bottom(sched_ids[-1])
    out_cy_target = span_top + (span_bot - span_top) / 2
    out = d.box(
        ox, out_cy_target - 58, out_w, 116,
        "MAC resource directives<br>active-UT mask<br>per-UT power<br>to the transmitter",
        "mac", fixed_width=True,
    )
    merge_x = cx + sched_w + gap // 2
    out_cy = d._center(out)[1]
    for cid in sched_ids:
        _bx, bcy = d._center(cid)
        d.route(cid, out, [(merge_x, bcy), (merge_x, out_cy)])

    ny = max(span_bot, d.bottom(out)) + row_gap + 10
    n1 = d.footnote(
        mx, ny, cw, 48,
        "Every scheduler reads the same adaptive-estimator feedback for a given batch (P2).",
        "neutral",
    )
    d.footnote(
        mx, d.bottom(n1) + 12, cw, 48,
        "The concurrency limit is two of the four terminals; static scheduling ignores it "
        "and keeps all four active.",
        "accent",
    )
    d.page_h = max(ph, int(d.bottom(n1) + 140))
    return d.to_xml()


def build_drl_rm_architecture() -> str:
    """Portrait canvas; the actor network shared by both learned schedulers."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    trunk_w = 460
    tx = mx + (cw - trunk_w) // 2

    d.title(mx, 16, cw, "Learned scheduler architecture")
    d.subtitle(
        mx,
        52,
        cw,
        "One actor-critic network; two PPO policies differing only in reward and weights.",
    )

    y = 112
    layers = [
        ("Policy state from the estimated channel<br>4 terminals × 130 features", "env"),
        ("Layer normalisation", "mac"),
        ("Two 1-D convolutions<br>64 filters, kernel 3, ReLU", "mac"),
        ("Per-terminal dense layer<br>128 units, ReLU", "mac"),
    ]
    ids = []
    for label, role in layers:
        ids.append(d.box(tx, y, trunk_w, 56, label, role, max_chars=40,
                         learned=(role == "mac")))
        y = d.bottom(ids[-1]) + row_gap

    ctx = d.box(
        tx, y, trunk_w, 56,
        "Global average pooling<br>context dense 128, ReLU",
        "mac", max_chars=40, learned=True,
    )
    y = d.bottom(ctx) + row_gap
    join = d.box(
        tx, y, trunk_w, 56,
        "Concatenate per-terminal<br>and context features",
        "mac", max_chars=40, learned=True,
    )
    y = d.bottom(join) + row_gap + 10

    head_gap = 14
    head_w = (trunk_w - 2 * head_gap) // 3
    h1 = d.box(tx, y, head_w, 76, "Scheduling logits<br>linear", "mac", fixed_width=True, learned=True)
    h2 = d.box(tx + head_w + head_gap, y, head_w, 76, "Power logits<br>linear", "mac", fixed_width=True, learned=True)
    h3 = d.box(tx + 2 * (head_w + head_gap), y, head_w, 76, "State value<br>(PPO critic)", "training", fixed_width=True)
    d.equalize_heights([h1, h2, h3])
    y = d.bottom(h1) + row_gap + 10

    proj = d.box(
        tx, y, trunk_w, 60,
        "Action: two terminals,<br>each at full or half power",
        "mac", fixed_width=True, max_chars=40,
    )
    y = d.bottom(proj) + row_gap
    out = d.box(
        tx + 40, y, trunk_w - 80, 56,
        "MAC resource directives<br>active-UT mask, per-UT power",
        "mac", max_chars=40,
    )

    for a, b in zip(ids[:-1], ids[1:]):
        d.edge(a, b)
    d.edge(ids[-1], ctx)
    d.edge(ctx, join)
    fan_y = d._geom[h1][1] - 10
    join_cx = d._center(join)[0]
    for head in (h1, h2):
        hx, _hy = d._center(head)
        d.route(join, head, [(join_cx, fan_y), (hx, fan_y)])
    # The value head reads the pooled context vector, not the joint features
    # (Dense(1)(context) in factory6g.models.drl_policy.create_policy_model).
    side_x = tx + trunk_w + 22
    ctx_cy = d._center(ctx)[1]
    h3x, _h3y = d._center(h3)
    d.route(ctx, h3, [(side_x, ctx_cy), (side_x, fan_y), (h3x, fan_y)])
    merge_y = d._geom[proj][1] - 10
    proj_cx = d._center(proj)[0]
    for head in (h1, h2):
        hx, _hy = d._center(head)
        d.route(head, proj, [(hx, merge_y), (proj_cx, merge_y)])
    d.edge(proj, out)

    ny = d.bottom(out) + row_gap + 10
    n1 = d.footnote(
        mx, ny, cw, 48,
        "The value head is the PPO critic, used only in training; evaluation reads the two logit heads.",
        "neutral",
    )
    d.footnote(
        mx, d.bottom(n1) + 12, cw, 48,
        "Both policies are trained by PPO against the simulator; "
        "the evaluation runs the fixed trained policy.",
        "accent",
    )
    d.page_h = max(ph, int(d.bottom(n1) + 140))
    return d.to_xml()


def build_lr_eval_coupling() -> str:
    """Landscape canvas; three-column literature evaluation-coupling patterns (Ch2 §2.7)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    col_gap = 64
    n_cols = 3
    col_w = (cw - (n_cols - 1) * col_gap) // n_cols
    row_gap = 24
    inner_pad = 18
    inner_gap = 14
    title_row_h = max(36, int(FONT_TITLE * 1.6) + 8)
    before_flow_gap = 20
    box_h = 72

    d.title(mx, 16, cw, "How prior CE and RM work is evaluated")
    d.subtitle(
        mx,
        52,
        cw,
        "Three recurring patterns in the channel-estimation and resource-management literature",
    )

    columns: list[dict[str, Any]] = [
        {
            "title": "PHY-only CE studies",
            "subtitle": "Classical and learned estimators (§2.4)",
            "flow": [
                ("Pilots", "neutral"),
                ("Estimator", "primary"),
                ("BER /<br>MSE", "success"),
            ],
            "footer": "No scheduler feedback",
            "footer_role": "primary",
        },
        {
            "title": "MAC-only RM studies",
            "subtitle": "Heuristic and learned schedulers (§2.5)",
            "flow": [
                ("Channel", "neutral"),
                ("Scheduler", "warning"),
                ("Rate /<br>fairness", "accent"),
            ],
            "footer": "Perfect CSI assumed",
            "footer_role": "warning",
        },
        {
            "title": "Coupled CE + RM",
            "subtitle": "The few joint studies",
            "flow": [
                ("Channel<br>draw", "primary"),
                ("Estimator<br>feeds RM", "secondary"),
                ("End-to-end<br>BER", "success"),
            ],
            "footer": "Rarely on factory channels",
            "footer_role": "secondary",
            "sparse": True,
        },
    ]

    panel_y = 108
    col_x = [mx + i * (col_w + col_gap) for i in range(n_cols)]
    inner_w = col_w - 2 * inner_pad
    bw = (inner_w - 2 * inner_gap) // 3

    panel_bottom = panel_y
    flow_ids: list[list[str]] = []

    for i, col in enumerate(columns):
        cx = col_x[i]
        ty = panel_y + inner_pad
        d.title(cx + inner_pad, ty, inner_w, col["title"])
        sty = ty + title_row_h
        sub_norm = normalize_label(col["subtitle"], max_chars=LABEL_MAX_CHARS_SUBTITLE)
        sub_lines = max(1, len(sub_norm.split("<br>")))
        sub_h = sub_lines * int(FONT_SUBTITLE * 1.4) + 8
        d.subtitle(cx + inner_pad, sty, inner_w, col["subtitle"])

        fy = sty + sub_h + before_flow_gap
        boxes: list[str] = []
        for j, (label, role) in enumerate(col["flow"]):
            bx = cx + inner_pad + j * (bw + inner_gap)
            bid = d.box(bx, fy, bw, box_h, label, role, fixed_width=True)
            boxes.append(bid)
            if j > 0:
                d.edge(boxes[j - 1], bid)
        flow_ids.append(boxes)

        ny = d.bottom(boxes[-1]) + row_gap
        nid = d.note(
            cx + inner_pad,
            ny,
            inner_w,
            48,
            col["footer"],
            col["footer_role"],
        )
        panel_bottom = max(panel_bottom, d.bottom(nid) + inner_pad)

    for i in range(n_cols):
        d.group_rect(col_x[i], panel_y, col_w, panel_bottom - panel_y)

    _, flow_cy = d._center(flow_ids[0][1])
    gap1_cx = col_x[0] + col_w + col_gap / 2
    gap2_cx = col_x[1] + col_w + col_gap / 2
    d.center_text(gap1_cx - 36, flow_cy - 22, 72, 44, "broken<br>link", color="#be123c")
    d.center_text(gap2_cx - 28, flow_cy - 16, 56, 32, "rare")

    # Closing claim is carried by the LaTeX caption and Table 2.1; no bottom banner.
    d.page_h = max(ph, int(panel_bottom + 96))
    return d.to_xml()


def build_decoupled_vs_integrated() -> str:
    """Landscape canvas; decoupled literature (left) vs staged integrated design (right)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    col_gap = 28
    panel_w = (cw - col_gap) // 2
    lx, rx = mx, mx + panel_w + col_gap
    row_gap = 14

    d.title(mx, 16, cw, "Decoupled versus integrated evaluation")
    d.subtitle(
        mx,
        52,
        cw,
        "Common PHY and MAC splits (left) against the staged coupled design (right).",
    )

    ly = 108
    inner_gap = 16
    inner_mx = lx + 18
    inner_w = panel_w - 36
    bw = (inner_w - 2 * inner_gap) // 3
    y1 = ly + 52

    d.heading(lx + 14, ly + 10, panel_w - 28, "Typical decoupled evaluation")
    d.heading(rx + 14, ly + 10, panel_w - 28, "Integrated design of this thesis")

    ce_ch = d.box(inner_mx, y1, bw, 72, "Channel<br>draw", "env", dashed=True, fixed_width=True)
    ce_est = d.box(inner_mx + bw + inner_gap, y1, bw, 72, "Estimator<br>comparison", "phy", fixed_width=True)
    ce_met = d.box(inner_mx + 2 * (bw + inner_gap), y1, bw, 72, "Estimation<br>error only", "output", fixed_width=True)
    d.equalize_heights([ce_ch, ce_est, ce_met])
    d.edge(ce_ch, ce_est)
    d.edge(ce_est, ce_met)
    ce_note = d.note(inner_mx, d.bottom(ce_ch) + 10, inner_w, 40, "No MAC decision is applied", "warning")

    y2 = d.bottom(ce_note) + row_gap + 12
    rm_ch = d.box(inner_mx, y2, bw, 72, "Channel<br>model", "env", dashed=True, fixed_width=True)
    rm_sched = d.box(inner_mx + bw + inner_gap, y2, bw, 72, "Scheduler with<br>perfect CSI", "mac", fixed_width=True)
    rm_out = d.box(inner_mx + 2 * (bw + inner_gap), y2, bw, 72, "Sum rate<br>only", "output", fixed_width=True)
    d.equalize_heights([rm_ch, rm_sched, rm_out])
    d.edge(rm_ch, rm_sched, dashed=True)
    d.edge(rm_sched, rm_out, dashed=True)
    left_note = d.note(
        inner_mx,
        d.bottom(rm_out) + 12,
        inner_w,
        44,
        "Estimation error never reaches the scheduler",
        "neutral",
    )

    ix = rx + 18
    igap = 14
    row_bw = (inner_w - 2 * igap) // 3
    iy = ly + 52
    ch = d.box(ix, iy, row_bw, 72, "Shared channel<br>draw (P1)", "env", fixed_width=True)
    ce = d.box(ix + row_bw + igap, iy, row_bw, 72, "Estimator<br>stage", "phy", fixed_width=True)
    fb = d.box(ix + 2 * (row_bw + igap), iy, row_bw, 72, "Estimate and<br>error variance (P2)", "coupling", fixed_width=True)
    d.equalize_heights([ch, ce, fb])
    d.edge(ch, ce)
    d.edge(ce, fb)

    iy2 = d.bottom(ch) + 40
    rm = d.box(ix, iy2, row_bw, 72, "Scheduler<br>stage", "mac", fixed_width=True)
    rd = d.box(ix + row_bw + igap, iy2, row_bw, 72, "MAC resource<br>directives", "mac", dashed=True, fixed_width=True)
    met = d.box(ix + 2 * (row_bw + igap), iy2, row_bw, 72, "Decoded BER and<br>throughput (P3)", "output", fixed_width=True)
    d.equalize_heights([rm, rd, met])
    fb_cx, fb_cy = d._center(fb)
    rm_cx, rm_cy = d._center(rm)
    wrap_y = iy2 - 20
    d.route(fb, rm, [(fb_cx, wrap_y), (rm_cx, wrap_y)], thick=True)
    d.edge(rm, rd)
    d.edge(rd, met)

    right_note = d.note(
        ix,
        d.bottom(met) + 12,
        inner_w,
        44,
        "Within each stage, every method reads the same realisation",
        "neutral",
    )

    panel_bottom = max(d.bottom(left_note), d.bottom(right_note)) + 18
    d.group_rect(lx, ly, panel_w, panel_bottom - ly)
    d.group_rect(rx, ly, panel_w, panel_bottom - ly)

    ny = panel_bottom + row_gap + 14
    d.footnote(
        mx,
        ny,
        cw,
        52,
        "Both panels are measured the same way: factory channel scenarios (P4) and "
        "end-to-end reliability, throughput and latency (P5).",
        "neutral",
    )
    d.page_h = max(ph, int(ny + 110))
    return d.to_xml()


def build_shared_context() -> str:
    """Landscape; one shared channel context feeds every estimator and one receiver (P1)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    top = 20
    row_h, row_gap = 62, 14

    # Column 1: the shared channel context (environment), one per batch.
    cx, cw = 20, 300
    items = [
        "Channel frequency response",
        "Noise variance",
        "Two independent noise draws",
        "Source bits",
    ]
    item_ids = []
    y = top + 52
    for label in items:
        item_ids.append(d.box(cx + 16, y, cw - 32, row_h - 10, label, "env",
                              fixed_width=True, max_chars=30))
        y = d.bottom(item_ids[-1]) + 10
    ctx_bottom = d.bottom(item_ids[-1]) + 16
    ctx = d.group_rect(cx, top, cw, ctx_bottom - top, role="env")
    d.heading(cx + 14, top + 10, cw - 28, "Shared channel context")

    # Column 2: the five estimators, identical inputs.
    ex, ew = cx + cw + 70, 230
    names = [("LS", False), ("DFT", False), ("LMMSE", False), ("Adaptive", False), ("Neural", True)]
    est_ids = []
    span = ctx_bottom - top
    total = len(names) * row_h + (len(names) - 1) * row_gap
    y = top + (span - total) / 2
    for name, learned in names:
        est_ids.append(d.box(ex, y, ew, row_h, f"{name} estimator", "phy",
                             fixed_width=True, learned=learned))
        y = d.bottom(est_ids[-1]) + row_gap

    # Column 3: one receiver shared by all estimators; column 4: the metric.
    rx, rw = ex + ew + 70, 240
    mid = top + span / 2
    rcv = d.box(rx, mid - 70, rw, 140,
                "Same receiver for every estimator<br>LMMSE equaliser<br>soft demapper<br>LDPC decoder",
                "phy", fixed_width=True, max_chars=34)
    ox, ow = rx + rw + 60, 170
    r_cy = d._center(rcv)[1]
    out = d.box(ox, r_cy - 45, ow, 90, "Decoded BER<br>one value per<br>estimator", "output",
                fixed_width=True)
    d.move_y(out, r_cy - d._geom[out][3] / 2)

    fan_x = cx + cw + 35
    _gx, gy, _gw, gh = d._geom[ctx]
    ctx_mid = gy + gh / 2
    for eid in est_ids:
        _bx, by = d._center(eid)
        d.route(ctx, eid, [(fan_x, ctx_mid), (fan_x, by)])
    merge_x = ex + ew + 35
    r_mid = d._center(rcv)[1]
    for eid in est_ids:
        _bx, by = d._center(eid)
        d.route(eid, rcv, [(merge_x, by), (merge_x, r_mid)])
    d.edge(rcv, out)

    d.page_h = int(ctx_bottom + 140)
    return d.to_xml()


def build_research_framework() -> str:
    """Full-page portrait research framework.

    Three colour-coded vertical tracks (RQ1 estimator / RQ2 scheduler / RQ3
    joint reading) run through five phase rows: problem, research question,
    approach, finding, contribution. Box text is condensed to phrases so the
    structure -- not the prose -- carries the figure. Canvas aspect is tuned so
    ``width=\\linewidth, height=0.80\\textheight`` fills the figure page without
    colliding with the page number or caption.
    """
    pw, ph = 1230, 1180
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 18
    label_w = 128
    gap = 10
    content_x = mx + label_w + gap
    content_w = pw - content_x - mx
    col_gap = 16
    col_w = (content_w - 2 * col_gap) // 3
    mc = 18

    # Track colours carry meaning: RQ1 = estimation (PHY), RQ2 = scheduling (MAC),
    # RQ3 = the joint PHY-MAC reading (coupling); the contribution is joint too.
    track_role = ("phy", "mac", "coupling")
    track_arrow = ("phy", "mac", "coupling")

    top_margin = 12
    bottom_margin = 12
    gap_budget = 70
    row_gap = gap_budget / 5
    usable = ph - top_margin - bottom_margin - gap_budget
    h_title = usable * 0.085
    h_problem = usable * 0.135
    h_question = usable * 0.145
    h_approach = usable * 0.40
    h_finding = usable * 0.13
    h_contrib = usable * 0.105

    def col_x(i: int) -> float:
        return content_x + i * (col_w + col_gap)

    def force_height(node_ids: list[str], height: float) -> None:
        id_set = set(node_ids)
        for cid, data in d.cells:
            if cid in id_set and "h" in data:
                data["h"] = height
                x, y, w, _ = d._geom[cid]
                d._geom[cid] = (x, y, w, height)

    def label_row(y0: float, height: float, text: str) -> None:
        d.row_label(mx, y0, label_w, height, text)

    def cell(x: float, y: float, w: float, text: str, role: str, *, min_h: float) -> str:
        return d.box(x, y, w, min_h, text, role, fixed_width=True, max_chars=mc)

    # --- Header band: thesis title (plain navy text, no box) ---
    y = top_margin
    d.title(
        content_x, y, content_w,
        "Integrated Machine Learning for Channel Estimation and Resource "
        "Management in 6G Smart-Factory Wireless",
    )
    y += h_title

    # --- Row 1: Problem (one per track) ---
    y += row_gap
    problem_labels = [
        "Dynamic factory channels break classical estimator assumptions",
        "PHY and MAC evaluated in isolation; estimation error not propagated",
        "Decoupled evidence cannot confirm end-to-end URLLC reliability",
    ]
    probs = [
        cell(col_x(i), y, col_w, t, track_role[i], min_h=h_problem)
        for i, t in enumerate(problem_labels)
    ]
    force_height(probs, h_problem)
    label_row(y, h_problem, "Problem")

    # --- Row 2: Research question ---
    y = d.bottom(probs[0]) + row_gap
    question_labels = [
        "RQ1: Neural vs LS estimator: BER, point status, runtime",
        "RQ2: Eight schedulers on shared estimated CSI: throughput, latency, "
        "power, fairness",
        "RQ3: Does a joint reading change rankings or trade-offs?",
    ]
    quest = [
        cell(col_x(i), y, col_w, t, track_role[i], min_h=h_question)
        for i, t in enumerate(question_labels)
    ]
    force_height(quest, h_question)
    label_row(y, h_question, "Research<br>question")
    for i, (p, q) in enumerate(zip(probs, quest)):
        d.edge(p, q, color=track_arrow[i])

    # --- Row 3: Approach (three condensed steps per track, in a dashed group) ---
    y = d.bottom(quest[0]) + row_gap
    inner_pad = 12
    sub_gap = 9
    inner_w = col_w - 2 * inner_pad
    approach_steps = [
        [
            "Neural vs LS, then all five estimators",
            "Transceiver chain held fixed",
            "Report BER, point status, runtime",
        ],
        [
            "Eight schedulers, estimated CSI at MAC",
            "Rayleigh / Rician / UMi / ray-traced",
            "Report throughput, latency, power, fairness",
        ],
        [
            "Estimator × scheduler pairings",
            "vs PHY-only and MAC-only views",
            "Check ranking, trade-off, attribution shifts",
        ],
    ]
    sub_h = (h_approach - 2 * inner_pad - 2 * sub_gap) / 3
    appr_tops: list[str] = []
    appr_bottoms: list[str] = []
    for i, steps in enumerate(approach_steps):
        gx = col_x(i)
        sub_ids: list[str] = []
        for j, step in enumerate(steps):
            sy = y + inner_pad + j * (sub_h + sub_gap)
            sid = cell(gx + inner_pad, sy, inner_w, step, track_role[i], min_h=sub_h)
            sub_ids.append(sid)
        force_height(sub_ids, sub_h)
        for j, sid in enumerate(sub_ids):
            x0, _, w0, _ = d._geom[sid]
            yj = y + inner_pad + j * (sub_h + sub_gap)
            d._geom[sid] = (x0, yj, w0, sub_h)
            for cid, data in d.cells:
                if cid == sid:
                    data["y"] = yj
                    data["h"] = sub_h
        d.group_rect(gx, y, col_w, h_approach, role=track_role[i])
        appr_tops.append(sub_ids[0])
        appr_bottoms.append(sub_ids[-1])
    label_row(y, h_approach, "Approach")
    for i, (q, top_id) in enumerate(zip(quest, appr_tops)):
        d.edge(q, top_id, color=track_arrow[i])

    # --- Row 4: Finding ---
    y = y + h_approach + row_gap
    finding_labels = [
        "Relative BER, point status, and runtime: neural vs LS",
        "Scheduler trade-offs under estimated CSI",
        "Joint reading vs PHY-only / MAC-only readings",
    ]
    finds = [
        cell(col_x(i), y, col_w, t, track_role[i], min_h=h_finding)
        for i, t in enumerate(finding_labels)
    ]
    force_height(finds, h_finding)
    label_row(y, h_finding, "Finding")
    for i, (mb, f) in enumerate(zip(appr_bottoms, finds)):
        d.edge(mb, f, color=track_arrow[i])
    # The RQ3 joint reading consumes the RQ1 and RQ2 findings; that convergence
    # is shown by all three Finding boxes flowing into the single Contribution
    # box below, so no cross-links are drawn inside the Finding row (they would
    # overlap the adjacent boxes).

    # --- Row 5: Contribution (full width) ---
    y = d.bottom(finds[0]) + row_gap
    contrib = d.box(
        content_x, y, content_w, h_contrib,
        "Joint PHY/MAC evaluation methodology for machine-learning<br>"
        "channel estimation and resource management in 6G smart factories",
        "key", fixed_width=True, max_chars=70,
    )
    force_height([contrib], h_contrib)
    label_row(y, h_contrib, "Contribution")
    for i, f in enumerate(finds):
        d.edge(f, contrib, color=track_arrow[i])

    d.page_h = ph
    return d.to_xml()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate thesis architecture figures as draw.io XML.")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        help="Write only these figure stems (e.g. fig_lr_taxonomy), without .drawio suffix.",
    )
    args = parser.parse_args()

    # Chapter 2 literature figures (fig_lr_taxonomy, fig_lr_eval_coupling) moved to
    # matplotlib in Aug 2026 -- see factory6g.visualization.thesis_engagement_figures.
    # build_lr_taxonomy / build_lr_eval_coupling are kept for history but no longer
    # dispatched.
    BUILDERS = {
        "fig_integrated_pipeline.drawio": build_integrated_pipeline,
        "fig_phy_stack.drawio": build_phy_stack,
        "fig_monte_carlo_flow.drawio": build_monte_carlo,
        "fig_factory_topology.drawio": build_factory_topology,
        "fig_estimator_comparison.drawio": build_estimator_comparison,
        "fig_rm_comparison.drawio": build_rm_comparison,
        "fig_drl_rm_architecture.drawio": build_drl_rm_architecture,
        "fig_decoupled_vs_integrated.drawio": build_decoupled_vs_integrated,
        "fig_research_framework.drawio": build_research_framework,
        "fig_shared_context.drawio": build_shared_context,
    }

    if args.only:
        wanted = {f"{stem}.drawio" if not stem.endswith(".drawio") else stem for stem in args.only}
        missing = wanted - set(BUILDERS)
        if missing:
            raise SystemExit(f"Unknown figure stem(s): {', '.join(sorted(missing))}")
        outputs = {name: BUILDERS[name]() for name in wanted}
    else:
        outputs = {name: fn() for name, fn in BUILDERS.items()}

    for name, xml in outputs.items():
        path = FIGDIR / name
        path.write_text(xml, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
