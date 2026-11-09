"""Ray-traced path-gain map of the hall at terminal height (Sionna RT 2.x).

Kept as the entry point for a map without renders; the work lives in
:mod:`factory6g.viewer.scene`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from factory6g.viewer.scene import build_downlink_scene, radio_map_of


def compute_radio_map(manifest_path: Path, *, cell_size: float = 0.5, max_depth: int = 3, samples_per_tx: int = 2_000_000) -> dict[str, Any]:
    scene, hall = build_downlink_scene(manifest_path)
    _, summary = radio_map_of(scene, hall, cell_size=cell_size, max_depth=max_depth, samples_per_tx=samples_per_tx)
    return summary
