"""Ray-traced path-gain map of the hall at terminal height (Sionna RT 2.x).

The base station transmits and the map is taken on a horizontal plane at the
terminal height. By reciprocity this is the uplink path gain a terminal at that
cell would see, including the base-station panel's pattern.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_RAYTRACING_FILE = Path(__file__).resolve().parents[1] / "components" / "raytracing.py"


def _load_raytracing():
    # factory6g.components imports the TensorFlow PHY on package import; the
    # viz image has no TensorFlow, so load the scene builder module on its own.
    spec = importlib.util.spec_from_file_location("factory6g_raytracing_standalone", _RAYTRACING_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_radio_map(
    manifest_path: Path,
    *,
    cell_size: float = 0.5,
    max_depth: int = 3,
    samples_per_tx: int = 2_000_000,
) -> dict[str, Any]:
    import json

    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("llvm_ad_mono_polarized")
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter

    rt = _load_raytracing()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    hall_raw = manifest["hall"]
    tuple_fields = {"room_dimensions", "bs_position", "machines"}
    hall = rt.HallSpec(**{
        name: tuple(hall_raw[name]) if name in tuple_fields else hall_raw[name]
        for name in rt.HallSpec.__dataclass_fields__
    })
    scene = rt.build_scene(hall)
    scene.remove("BS")
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=hall.num_bs_ant // 2,
        vertical_spacing=hall.antenna_spacing, horizontal_spacing=hall.antenna_spacing,
        pattern=hall.bs_pattern, polarization="cross",
    )
    scene.add(Transmitter("BS", position=list(hall.bs_position), orientation=[0.0, math.pi / 2, 0.0]))

    length, width, _ = hall.room_dimensions
    radio_map = RadioMapSolver()(
        scene,
        center=[0.0, 0.0, float(hall.ut_height)],
        orientation=[0.0, 0.0, 0.0],
        size=[float(length), float(width)],
        cell_size=[cell_size, cell_size],
        max_depth=max_depth,
        samples_per_tx=samples_per_tx,
    )
    gain = np.asarray(radio_map.path_gain.numpy())[0]  # [cells_y, cells_x]
    centers = np.asarray(radio_map.cell_centers.numpy())  # [cells_y, cells_x, 3]
    # Read the grid layout from the solver rather than assuming it: rows must
    # step along +y and columns along +x for the viewer's plan.
    if not (centers[1, 0, 1] > centers[0, 0, 1] and centers[0, 1, 0] > centers[0, 0, 0]):
        raise ValueError("Unexpected radio-map grid orientation from Sionna RT.")
    with np.errstate(divide="ignore"):
        gain_db = 10.0 * np.log10(np.where(gain > 0, gain, np.nan))
    return {
        "source": "Sionna RT RadioMapSolver",
        "max_depth": max_depth,
        "samples_per_tx": samples_per_tx,
        "height_m": float(hall.ut_height),
        "cell_size_m": cell_size,
        "x0": float(centers[0, 0, 0]) - cell_size / 2,
        "y0": float(centers[0, 0, 1]) - cell_size / 2,
        # Row 0 is the lowest y; NaN cells (no path reached them) export as null.
        "gain_db": [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in gain_db],
    }
