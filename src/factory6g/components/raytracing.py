"""Site-specific ray-traced channels for the canonical factory hall.

The hall of Figures 3.15/3.16 (room, machine layout, materials, base-station
panel) is built as a Sionna RT scene and traced once, offline, for a set of
terminal positions. Each position yields one uplink frequency response from a
single-antenna terminal to the eight-port base station over the 128
subcarriers. The responses are stored as a *channel bank*; the Monte Carlo
pipeline then draws ``num_ut`` positions per slot from the bank instead of
re-tracing, so a ray-traced batch costs no more than a statistical one.

Positions are split into disjoint training and evaluation sets, so the neural
estimator and the DRL schedulers never see an evaluation position.

Conventions shared with the rest of the pipeline:

* Uplink: terminals transmit, the base station receives.
* BS array: 1 x (num_bs_ant/2) cross-polarised TR 38.901 elements, the same
  geometry as :class:`factory6g.components.antenna.AntennaConfig`; the panel
  hangs from the ceiling and faces the floor.
* Delays are normalised so the first path arrives at zero (ideal timing), and
  each link is power-normalised over antennas and subcarriers exactly as
  ``GenerateOFDMChannel(normalize_channel=True)`` does for the statistical
  models, so the operating point is set by Eb/N0 alone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BANK_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Hall geometry
# ---------------------------------------------------------------------------

def sample_machine_layout(
    room_dimensions: Sequence[float],
    *,
    num_machines: int,
    size_range: Sequence[Sequence[float]],
    padding: float,
    seed: int,
) -> list[dict[str, float]]:
    """Machine footprints in metres, centred on the hall origin.

    Draw order (size x, y, z, then position x, y) matches the thesis plan-view
    figure, so the same seed gives the same hall in the figure and the trace.
    """
    rng = np.random.default_rng(seed)
    room_len, room_width = float(room_dimensions[0]), float(room_dimensions[1])
    layout: list[dict[str, float]] = []
    for _ in range(int(num_machines)):
        sx = float(rng.uniform(size_range[0][0], size_range[0][1]))
        sy = float(rng.uniform(size_range[1][0], size_range[1][1]))
        sz = float(rng.uniform(size_range[2][0], size_range[2][1]))
        px = float(rng.uniform(-room_len / 2 + padding, room_len / 2 - padding))
        py = float(rng.uniform(-room_width / 2 + padding, room_width / 2 - padding))
        layout.append({"x": px, "y": py, "z": 0.0, "sx": sx, "sy": sy, "sz": sz})
    return layout


def sample_terminal_positions(
    rng: np.random.Generator,
    count: int,
    *,
    room_dimensions: Sequence[float],
    boundary_padding: float,
    height: float,
    machines: Sequence[dict[str, float]],
    clearance: float = 0.3,
) -> np.ndarray:
    """Uniform terminal positions on the floor, outside every machine footprint.

    Returns an array of shape ``[count, 3]``.
    """
    room_len, room_width = float(room_dimensions[0]), float(room_dimensions[1])
    out = np.empty((count, 3), dtype=np.float64)
    filled = 0
    while filled < count:
        x = rng.uniform(-room_len / 2 + boundary_padding, room_len / 2 - boundary_padding)
        y = rng.uniform(-room_width / 2 + boundary_padding, room_width / 2 - boundary_padding)
        inside = any(
            abs(x - m["x"]) <= m["sx"] / 2 + clearance and abs(y - m["y"]) <= m["sy"] / 2 + clearance
            for m in machines
        )
        if inside:
            continue
        out[filled] = (x, y, height)
        filled += 1
    return out


def checkerboard_split(
    positions: np.ndarray,
    *,
    cell: float,
    guard: float,
) -> np.ndarray:
    """Assign positions to a spatial split: 0 = train, 1 = eval, -1 = guard band.

    The floor is tiled into ``cell``-metre squares coloured like a checkerboard;
    positions within ``guard`` metres of a cell edge are discarded, so every
    evaluation position is at least ``2 * guard`` from every training position
    (several wavelengths at 3.5 GHz) and small-scale fading is decorrelated.
    """
    x, y = positions[:, 0], positions[:, 1]
    fx, fy = np.mod(x, cell), np.mod(y, cell)
    in_guard = (fx < guard) | (fx > cell - guard) | (fy < guard) | (fy > cell - guard)
    parity = (np.floor(x / cell) + np.floor(y / cell)).astype(int) % 2
    return np.where(in_guard, -1, parity)


def _scene_xml(
    room_dimensions: Sequence[float],
    machines: Sequence[dict[str, float]],
    *,
    wall_thickness: float,
    box_ply_path: Path,
) -> str:
    # Sionna's box.ply spans x, y in [-5, 5] and z in [0, 5]: scaling by size/10
    # (x, y) and size/5 (z) gives a box of that size centred in x-y with its
    # base at the translated z.
    lines = [
        '<scene version="2.1.0">',
        '    <bsdf type="itu-radio-material" id="mat_default"><string name="type" value="concrete"/></bsdf>',
    ]

    def add(shape_id: str, size: Sequence[float], base: Sequence[float]) -> None:
        lines.extend([
            f'    <shape type="ply" id="{shape_id}">',
            f'        <string name="filename" value="{box_ply_path}"/>',
            '        <ref id="mat_default" name="bsdf"/>',
            '        <transform name="to_world">',
            f'            <scale x="{size[0] / 10.0}" y="{size[1] / 10.0}" z="{size[2] / 5.0}"/>',
            f'            <translate x="{base[0]}" y="{base[1]}" z="{base[2]}"/>',
            '        </transform>',
            '    </shape>',
        ])

    length, width, height = (float(v) for v in room_dimensions)
    t = float(wall_thickness)
    add("floor", [length, width, t], [0.0, 0.0, -t])
    add("ceiling", [length, width, t], [0.0, 0.0, height])
    add("wall_left", [t, width, height], [-length / 2 - t / 2, 0.0, 0.0])
    add("wall_right", [t, width, height], [length / 2 + t / 2, 0.0, 0.0])
    add("wall_front", [length, t, height], [0.0, -width / 2 - t / 2, 0.0])
    add("wall_back", [length, t, height], [0.0, width / 2 + t / 2, 0.0])
    for i, m in enumerate(machines):
        add(f"machine_{i}", [m["sx"], m["sy"], m["sz"]], [m["x"], m["y"], 0.0])
    lines.append("</scene>")
    return "\n".join(lines)


@dataclass(frozen=True)
class HallSpec:
    """Everything that defines the traced hall; recorded in the bank manifest."""

    room_dimensions: tuple[float, float, float]
    machines: tuple[dict[str, float], ...]
    wall_thickness: float
    metal: dict[str, Any]
    concrete: dict[str, Any]
    carrier_frequency: float
    num_bs_ant: int
    bs_position: tuple[float, float, float]
    ut_height: float
    antenna_spacing: float
    bs_pattern: str
    ut_pattern: str
    ut_polarization: str

    @classmethod
    def from_config(cls, raw: dict[str, Any], *, layout_seed: int) -> "HallSpec":
        """Build from the raw (sectioned) config.json dictionary."""
        fs = raw["factory_scenario"]
        system = raw["system"]
        tr = raw["transceiver"]
        room = tuple(float(v) for v in fs["room_dimensions"])
        machines = sample_machine_layout(
            room,
            num_machines=int(fs["num_machines"]),
            size_range=fs["machine_size_range"],
            padding=float(tr.get("room_padding", 2.0)),
            seed=layout_seed,
        )
        return cls(
            room_dimensions=room,  # type: ignore[arg-type]
            machines=tuple(machines),
            wall_thickness=float(tr.get("wall_thickness", 0.2)),
            metal=dict(fs["materials"]["metal"]),
            concrete=dict(fs["materials"]["concrete"]),
            carrier_frequency=float(system["carrier_frequency"]),
            num_bs_ant=int(system["num_bs_ant"]),
            bs_position=(0.0, 0.0, room[2] - float(tr.get("tx_height_offset", 1.0))),
            ut_height=float(tr.get("rx_height", 1.0)),
            antenna_spacing=float(tr.get("antenna_spacing", 0.5)),
            bs_pattern=str(tr.get("tx_pattern", "tr38901")),
            ut_pattern=str(tr.get("rx_pattern", "iso")),
            ut_polarization=str(tr.get("rx_polarization", "V")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_dimensions": list(self.room_dimensions),
            "machines": [dict(m) for m in self.machines],
            "wall_thickness": self.wall_thickness,
            "metal": self.metal,
            "concrete": self.concrete,
            "carrier_frequency": self.carrier_frequency,
            "num_bs_ant": self.num_bs_ant,
            "bs_position": list(self.bs_position),
            "bs_array": f"1x{self.num_bs_ant // 2} cross-polarised, facing floor",
            "ut_height": self.ut_height,
            "antenna_spacing": self.antenna_spacing,
            "bs_pattern": self.bs_pattern,
            "ut_pattern": self.ut_pattern,
            "ut_polarization": self.ut_polarization,
        }


def build_scene(hall: HallSpec):
    """Sionna RT scene of the hall with the base station as the receiver."""
    import tempfile

    from sionna.rt import PlanarArray, RadioMaterial, Receiver, load_scene
    from sionna.rt.scene import box as box_xml_path

    box_ply = Path(box_xml_path).parent / "meshes" / "box.ply"
    xml = _scene_xml(hall.room_dimensions, hall.machines, wall_thickness=hall.wall_thickness, box_ply_path=box_ply)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(xml)
        xml_path = Path(handle.name)
    try:
        scene = load_scene(str(xml_path))
    finally:
        xml_path.unlink(missing_ok=True)

    metal = RadioMaterial(
        str(hall.metal.get("name", "factory_metal")),
        relative_permittivity=float(hall.metal["relative_permittivity"]),
        conductivity=float(hall.metal["conductivity"]),
    )
    concrete = RadioMaterial(
        str(hall.concrete.get("name", "factory_concrete")),
        relative_permittivity=float(hall.concrete["relative_permittivity"]),
        conductivity=float(hall.concrete["conductivity"]),
    )
    scene.add(metal)
    scene.add(concrete)
    for name, obj in scene.objects.items():
        obj.radio_material = metal if name.startswith("machine") else concrete

    scene.frequency = hall.carrier_frequency
    # Uplink: terminals are the transmitters, the base station is the receiver.
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        pattern=hall.ut_pattern, polarization=hall.ut_polarization,
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=hall.num_bs_ant // 2,
        vertical_spacing=hall.antenna_spacing, horizontal_spacing=hall.antenna_spacing,
        pattern=hall.bs_pattern, polarization="cross",
    )
    # Boresight (+x) rotated about y by +90 deg points at -z: the ceiling panel
    # faces the floor.
    scene.add(Receiver("BS", position=list(hall.bs_position), orientation=[0.0, math.pi / 2, 0.0]))
    return scene


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    h: np.ndarray                 # [N, num_bs_ant, F] complex64, unnormalised
    los: np.ndarray               # [N] bool, direct path present
    k_factor_db: np.ndarray       # [N] float32, strongest-path / rest power (dB)
    rms_delay_spread_ns: np.ndarray  # [N] float32
    num_paths: np.ndarray         # [N] int32


def trace_positions(
    scene,
    positions: np.ndarray,
    frequencies: np.ndarray,
    *,
    max_depth: int,
    samples_per_src: int,
    chunk: int = 64,
    seed: int = 0,
) -> TraceResult:
    """Trace every position as a transmitter to the BS and return CFRs and stats."""
    import drjit as dr
    from sionna.rt import PathSolver, Transmitter

    solver = PathSolver()
    freqs = dr.llvm.ad.Float(np.asarray(frequencies, dtype=np.float32))
    hs, los_all, k_all, ds_all, np_all = [], [], [], [], []
    for start in range(0, len(positions), chunk):
        block = positions[start:start + chunk]
        names = [f"ut{start + i}" for i in range(len(block))]
        for name, pos in zip(names, block):
            scene.add(Transmitter(name, position=[float(v) for v in pos]))
        try:
            paths = solver(
                scene,
                max_depth=max_depth,
                samples_per_src=samples_per_src,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
                seed=seed + start,
            )
            # cfr: [num_rx=1, rx_ant, num_tx, tx_ant=1, time=1, F]
            h = paths.cfr(freqs, normalize_delays=True, normalize=False, out_type="numpy")
            hs.append(np.transpose(h[0, :, :, 0, 0, :], (1, 0, 2)).astype(np.complex64))

            a, tau = paths.cir(normalize_delays=True, out_type="numpy")
            # a: [1, rx_ant, num_tx, 1, P, 1]; tau: [1, num_tx, P]
            power = np.sum(np.abs(a[0, :, :, 0, :, 0]) ** 2, axis=0)  # [num_tx, P]
            delays = tau[0]
            interactions = np.asarray(paths.interactions.numpy())  # [depth, 1, num_tx, P]
            valid = np.asarray(paths.valid.numpy())[0]              # [num_tx, P]
            direct = np.all(interactions[:, 0] == 0, axis=0) & valid
            for t in range(len(block)):
                p = np.where(valid[t], power[t], 0.0)
                total = p.sum()
                los_all.append(bool(direct[t].any()))
                strongest = p.max() if total > 0 else 0.0
                rest = total - strongest
                k_all.append(10 * np.log10(strongest / rest) if rest > 0 else np.inf)
                if total > 0:
                    d = np.where(valid[t], delays[t], 0.0)
                    mean = np.sum(p * d) / total
                    ds_all.append(1e9 * math.sqrt(max(np.sum(p * d ** 2) / total - mean ** 2, 0.0)))
                else:
                    ds_all.append(0.0)
                np_all.append(int(valid[t].sum()))
        finally:
            for name in names:
                scene.remove(name)
    return TraceResult(
        h=np.concatenate(hs, axis=0),
        los=np.asarray(los_all, dtype=bool),
        k_factor_db=np.asarray(k_all, dtype=np.float32),
        rms_delay_spread_ns=np.asarray(ds_all, dtype=np.float32),
        num_paths=np.asarray(np_all, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Channel bank
# ---------------------------------------------------------------------------

def save_bank(path: Path, splits: dict[str, tuple[np.ndarray, TraceResult]], manifest: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split, (positions, result) in splits.items():
        arrays[f"{split}_h"] = result.h
        arrays[f"{split}_positions"] = positions.astype(np.float32)
        arrays[f"{split}_los"] = result.los
        arrays[f"{split}_k_factor_db"] = result.k_factor_db
        arrays[f"{split}_rms_delay_spread_ns"] = result.rms_delay_spread_ns
        arrays[f"{split}_num_paths"] = result.num_paths
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def subcarrier_frequencies(num_subcarriers: int, spacing: float) -> np.ndarray:
    """Baseband subcarrier frequencies, as sionna.phy.channel.subcarrier_frequencies (no TensorFlow)."""
    n = int(num_subcarriers)
    start = -(n // 2) if n % 2 == 0 else -((n - 1) // 2)
    return (np.arange(start, start + n, dtype=np.float64) * float(spacing)).astype(np.float32)


def tracer_versions() -> dict[str, str]:
    out = {}
    for name in ("sionna.rt", "mitsuba", "drjit"):
        try:
            module = __import__(name, fromlist=["__version__"])
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            pass
    return out


def load_bank_split(path: str | Path, split: str) -> np.ndarray:
    """Uplink frequency responses ``[N, num_bs_ant, F]`` for one split."""
    with np.load(path) as data:
        key = f"{split}_h"
        if key not in data:
            raise KeyError(f"Channel bank {path} has no split '{split}'.")
        return np.asarray(data[key], dtype=np.complex64)
