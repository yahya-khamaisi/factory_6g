"""Sionna RT 2.x scene of the hall for the viewer: radio map, per-link paths, renders.

One downlink scene serves everything: the ceiling base station transmits with
its 1 x 4 cross-polarised panel and receivers sit at the terminal positions of
one traced batch. By reciprocity the traced paths are the uplink paths the
simulation used, and the radio map is the uplink path gain a terminal would
see at each cell.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_RAYTRACING_FILE = Path(__file__).resolve().parents[1] / "components" / "raytracing.py"
INTERACTION_NAMES = {1: "specular", 2: "diffuse", 4: "refraction", 8: "diffraction"}
# Same propagation mechanisms as the channel bank the simulation draws from
# (data/rt_channel_bank.npz.manifest.json): LOS and specular reflection only.
BANK_MECHANISMS = {"los": True, "specular_reflection": True, "diffuse_reflection": False, "refraction": False}


def _load_raytracing():
    # factory6g.components imports the TensorFlow PHY on package import; the
    # viz image has no TensorFlow, so load the scene builder module on its own.
    spec = importlib.util.spec_from_file_location("factory6g_raytracing_standalone", _RAYTRACING_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_mitsuba() -> None:
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("llvm_ad_mono_polarized")


def build_downlink_scene(manifest_path: Path, terminal_positions: Sequence[Sequence[float]] = ()):
    """Hall scene with the base station transmitting and receivers at the given positions."""
    _init_mitsuba()
    from sionna.rt import PlanarArray, Receiver, Transmitter

    rt = _load_raytracing()
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["hall"]
    tuple_fields = {"room_dimensions", "bs_position", "machines"}
    hall = rt.HallSpec(**{
        name: tuple(raw[name]) if name in tuple_fields else raw[name]
        for name in rt.HallSpec.__dataclass_fields__
    })
    scene = rt.build_scene(hall)
    scene.remove("BS")
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=hall.num_bs_ant // 2,
        vertical_spacing=hall.antenna_spacing, horizontal_spacing=hall.antenna_spacing,
        pattern=hall.bs_pattern, polarization="cross",
    )
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern=hall.ut_pattern, polarization=hall.ut_polarization)
    # Boresight (+x) rotated about y by +90 deg points at -z: the panel faces the floor.
    scene.add(Transmitter("BS", position=list(hall.bs_position), orientation=[0.0, math.pi / 2, 0.0]))
    for i, pos in enumerate(terminal_positions):
        scene.add(Receiver(f"UT{i}", position=[float(v) for v in pos]))
    return scene, hall


def radio_map_of(scene, hall, *, cell_size: float = 0.5, max_depth: int = 3, samples_per_tx: int = 2_000_000):
    from sionna.rt import RadioMapSolver

    length, width, _ = hall.room_dimensions
    radio_map = RadioMapSolver()(
        scene,
        center=[0.0, 0.0, float(hall.ut_height)],
        orientation=[0.0, 0.0, 0.0],
        size=[float(length), float(width)],
        cell_size=[cell_size, cell_size],
        max_depth=max_depth,
        samples_per_tx=samples_per_tx,
        **BANK_MECHANISMS,
    )
    gain = np.asarray(radio_map.path_gain.numpy())[0]  # [cells_y, cells_x]
    centers = np.asarray(radio_map.cell_centers.numpy())  # [cells_y, cells_x, 3]
    # Read the grid layout from the solver rather than assuming it: rows must
    # step along +y and columns along +x for the viewer's plan.
    if not (centers[1, 0, 1] > centers[0, 0, 1] and centers[0, 1, 0] > centers[0, 0, 0]):
        raise ValueError("Unexpected radio-map grid orientation from Sionna RT.")
    with np.errstate(divide="ignore"):
        gain_db = 10.0 * np.log10(np.where(gain > 0, gain, np.nan))
    summary = {
        "source": "Sionna RT RadioMapSolver, LOS + specular",
        "max_depth": max_depth,
        "samples_per_tx": samples_per_tx,
        "height_m": float(hall.ut_height),
        "cell_size_m": cell_size,
        "x0": float(centers[0, 0, 0]) - cell_size / 2,
        "y0": float(centers[0, 0, 1]) - cell_size / 2,
        # Row 0 is the lowest y; NaN cells (no path reached them) export as null.
        "gain_db": [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in gain_db],
    }
    return radio_map, summary


def _path_label(types: Sequence[int]) -> str:
    seq = [INTERACTION_NAMES.get(int(t), f"type {int(t)}") for t in types if int(t) != 0]
    if not seq:
        return "LOS"
    if len(set(seq)) == 1:
        return seq[0] if len(seq) == 1 else f"{seq[0]} ×{len(seq)}"
    return " → ".join(seq)


def link_details(paths, hall, *, num_subcarriers: int, subcarrier_spacing: float, carrier_frequency: float, top: int = 8) -> list[dict[str, Any]]:
    """Strongest paths, delay spread, K-factor and |H(f)| per port for every terminal."""
    a_re, a_im = paths.a
    a = np.asarray(a_re.numpy()) + 1j * np.asarray(a_im.numpy())  # [rx, rx_ant, tx, tx_ant, paths]
    tau = np.asarray(paths.tau.numpy())  # [rx, tx, paths] (synthetic array)
    inter = np.asarray(paths.interactions.numpy())  # [depth, rx, tx, paths]
    verts = np.asarray(paths.vertices.numpy())  # [depth, rx, tx, paths, 3]
    valid = np.asarray(paths.valid.numpy()).astype(bool)  # [rx, tx, paths]

    freqs = carrier_frequency + (np.arange(num_subcarriers) - num_subcarriers // 2) * subcarrier_spacing
    import drjit as dr

    h = paths.cfr(frequencies=dr.llvm.ad.Float(freqs.astype(np.float32)), normalize_delays=True, out_type="numpy")
    h = np.asarray(h)  # [rx, rx_ant, tx, tx_ant, time, freq]

    links = []
    bs = np.asarray(hall.bs_position, dtype=float)
    for u in range(a.shape[0]):
        ok = valid[u, 0]
        power = np.sum(np.abs(a[u, :, 0, :, :]) ** 2, axis=(0, 1))  # per path, over ports
        idx = np.flatnonzero(ok & (power > 0))
        if idx.size == 0:
            links.append({"terminal": u, "num_paths": 0})
            continue
        p = power[idx]
        t = tau[u, 0, idx]
        t_rel = t - t.min()
        mean_tau = float(np.sum(p * t_rel) / p.sum())
        rms_ns = float(np.sqrt(np.sum(p * (t_rel - mean_tau) ** 2) / p.sum()) * 1e9)
        is_los = np.all(inter[:, u, 0, idx] == 0, axis=0)
        k_db = float(10 * np.log10(p[is_los].sum() / p[~is_los].sum())) if is_los.any() and (~is_los).any() else None
        order = np.argsort(p)[::-1][:top]
        strongest = float(p.max())
        rows = []
        for j in order:
            k = idx[j]
            types = inter[:, u, 0, k]
            pts = [verts[d, u, 0, k].tolist() for d in range(types.size) if int(types[d]) != 0]
            rows.append({
                "type": _path_label(types),
                "delay_ns": round(float(t_rel[j]) * 1e9, 2),
                "power_db": round(10 * float(np.log10(p[j] / strongest)), 2),
                "vertices": [[round(v, 3) for v in pt] for pt in pts],
            })
        mag_db = 10 * np.log10(np.maximum(np.abs(h[u, 0, 0, :, 0, :]) ** 2, 1e-30))  # [ports, freq]
        links.append({
            "terminal": u,
            "num_paths": int(idx.size),
            "rms_delay_spread_ns": round(rms_ns, 2),
            "k_factor_db": None if k_db is None else round(k_db, 2),
            "los": bool(is_los.any()),
            "distance_m": None,
            "paths": rows,
            "cfr_db": [[round(float(v), 2) for v in port[::2]] for port in mag_db],
            "bs_position": bs.tolist(),
        })
    return links


def render_views(scene, *, paths, radio_map, out_dir: Path, hall, num_samples: int = 128) -> list[dict[str, str]]:
    """Two stills with the line-of-sight and single-bounce paths of every terminal."""
    from sionna.rt import Camera

    L, W, H = hall.room_dimensions
    # Steep enough to look over the near walls into the hall.
    views = [
        ("render_iso.png", "Corner view, ceiling removed", Camera(position=[-0.75 * L, -0.75 * W, 3.4 * H], look_at=[0.0, 0.0, 0.0])),
        ("render_top.png", "Top view, ceiling removed", Camera(position=[0.0, -0.01, 2.7 * max(L, W)], look_at=[0.0, 0.0, 0.0])),
    ]
    written = []
    for filename, label, camera in views:
        scene.render_to_file(
            camera=camera, filename=str(Path(out_dir) / filename), clip_at=float(H) - 0.3,
            paths=paths, radio_map=radio_map, rm_cmap="Blues", resolution=(1280, 800), num_samples=num_samples,
            show_orientations=False,
        )
        written.append({"file": filename, "label": label})
    return written


def compute_scene(
    manifest_path: Path,
    terminal_positions: Sequence[Sequence[float]],
    *,
    out_dir: Path,
    num_subcarriers: int = 128,
    subcarrier_spacing: float = 30e3,
    max_depth: int = 5,
) -> dict[str, Any]:
    """Radio map, per-link details and two renders for one batch's terminal positions."""
    from sionna.rt import PathSolver

    scene, hall = build_downlink_scene(manifest_path, terminal_positions)
    radio_map, rm_summary = radio_map_of(scene, hall, max_depth=max_depth)
    paths = PathSolver()(scene, max_depth=max_depth, samples_per_src=200_000, synthetic_array=True, **BANK_MECHANISMS)
    # Drawing every path buries the hall; the stills show LOS and first-order reflections.
    render_paths = PathSolver()(scene, max_depth=1, samples_per_src=200_000, synthetic_array=True, **BANK_MECHANISMS)
    links = link_details(paths, hall, num_subcarriers=num_subcarriers, subcarrier_spacing=subcarrier_spacing,
                         carrier_frequency=float(hall.carrier_frequency))
    for link, pos in zip(links, terminal_positions):
        link["distance_m"] = round(float(np.linalg.norm(np.asarray(pos) - np.asarray(hall.bs_position))), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders = render_views(scene, paths=render_paths, radio_map=radio_map, out_dir=out_dir, hall=hall)
    return {"radio_map": rm_summary, "links": links, "renders": renders, "max_depth": max_depth}
