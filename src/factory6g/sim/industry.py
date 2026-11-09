"""Industry 5.0 scenario layer (Phase 2): device classes, body blockage, disruptions, power caps.

Optional and off by default; enabled by an ``industry`` section in the config.

* **Device classes.** Each terminal belongs to a class with a latency budget,
  a reliability target and a survival time, counted in scheduling intervals
  (one batch = one interval). The targets are set in the config; the catalog
  values follow the use-case classes of 3GPP TS 22.104 in spirit but are
  scenario settings, not quoted from the specification.
* **Body blockage.** Body-worn devices are shadowed by their wearer in a share
  of batches, with a fixed extra attenuation. This is a deliberately simple
  stand-in for the TR 38.901 §7.6.4 blockage models: one Bernoulli draw per
  terminal and batch, from a generator seeded by the run seed and the batch
  index, so it never consumes draws from the shared channel and noise
  generators.
* **Disruptions.** Scripted events over batch ranges: ``bs_port_fault``
  (listed base-station ports carry no signal), ``ut_blockage`` (extra
  attenuation on one terminal) and ``interference_burst`` (an SINR loss on
  every link, applied as a channel attenuation).
* **Exposure-aware power.** Body-worn classes have a power cap (fraction of the
  maximum UE power) that is enforced on every scheduler's directives.

All channel changes are applied inside ``Model.prepare_batch_context`` before
the scheduler feedback is formed, so schedulers see the disruption as a real
system would.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

DEVICE_CLASSES: dict[str, dict[str, Any]] = {
    "cobot_motion": {"label": "Cobot motion control", "latency_intervals": 1, "reliability": 0.999999, "survival_intervals": 1, "body_worn": False, "power_cap": 1.0},
    "agv_control": {"label": "Mobile robot control", "latency_intervals": 2, "reliability": 0.99999, "survival_intervals": 1, "body_worn": False, "power_cap": 1.0},
    "safety_wearable": {"label": "Worker safety wearable", "latency_intervals": 1, "reliability": 0.99999, "survival_intervals": 1, "body_worn": True, "power_cap": 0.5},
    "ar_assist": {"label": "AR assistance", "latency_intervals": 2, "reliability": 0.999, "survival_intervals": 2, "body_worn": True, "power_cap": 0.5},
    "condition_sensor": {"label": "Condition monitoring", "latency_intervals": 4, "reliability": 0.999, "survival_intervals": 4, "body_worn": False, "power_cap": 1.0},
}
EVENT_TYPES = {"bs_port_fault", "ut_blockage", "interference_burst"}


@dataclass(frozen=True)
class IndustryConfig:
    device_classes: list[str]
    body_blockage_probability: float = 0.0
    body_blockage_db: float = 20.0
    events: list[dict[str, Any]] = field(default_factory=list)
    class_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndustryConfig":
        allowed = {"device_classes", "body_blockage", "events", "class_overrides"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in 'industry': {sorted(unknown)}.")
        classes = [str(c) for c in raw.get("device_classes", [])]
        overrides = {str(k): dict(v) for k, v in raw.get("class_overrides", {}).items()}
        for name in classes:
            if name not in DEVICE_CLASSES and name not in overrides:
                raise ValueError(f"Unknown device class '{name}'. Known: {sorted(DEVICE_CLASSES)}.")
        blockage = dict(raw.get("body_blockage", {}))
        events = [dict(e) for e in raw.get("events", [])]
        for e in events:
            if e.get("type") not in EVENT_TYPES:
                raise ValueError(f"Unknown event type {e.get('type')!r}. Known: {sorted(EVENT_TYPES)}.")
            if int(e.get("start", -1)) < 0 or int(e.get("end", -1)) < int(e.get("start", 0)):
                raise ValueError(f"Event {e} needs 0 <= start <= end (batch indices, inclusive).")
        return cls(
            device_classes=classes,
            body_blockage_probability=float(blockage.get("probability", 0.0)),
            body_blockage_db=float(blockage.get("attenuation_db", 20.0)),
            events=events,
            class_overrides=overrides,
        )

    def class_spec(self, name: str) -> dict[str, Any]:
        return {**DEVICE_CLASSES.get(name, {}), **self.class_overrides.get(name, {})}


class IndustryScenario:
    def __init__(self, config: IndustryConfig, *, num_ut: int, num_bs_ant: int, seed: int) -> None:
        if len(config.device_classes) != num_ut:
            raise ValueError(f"industry.device_classes lists {len(config.device_classes)} classes for {num_ut} terminals.")
        self.config = config
        self.num_ut = int(num_ut)
        self.num_bs_ant = int(num_bs_ant)
        self.seed = int(seed)
        self.specs = [config.class_spec(c) for c in config.device_classes]
        self.body_worn = np.array([bool(s.get("body_worn")) for s in self.specs])
        self.power_cap = np.array([float(s.get("power_cap", 1.0)) for s in self.specs])

    def meta(self) -> dict[str, Any]:
        return {"device_classes": self.config.device_classes, "class_specs": self.specs,
                "body_blockage": {"probability": self.config.body_blockage_probability, "attenuation_db": self.config.body_blockage_db},
                "events": self.config.events}

    def batch_state(self, batch_index: int) -> dict[str, Any]:
        rng = np.random.default_rng([self.seed, int(batch_index), 7])
        blocked = (rng.random(self.num_ut) < self.config.body_blockage_probability) & self.body_worn
        att = np.where(blocked, self.config.body_blockage_db, 0.0)
        ports = np.ones(self.num_bs_ant)
        sinr_loss = 0.0
        active = []
        for e in self.config.events:
            if not (int(e["start"]) <= batch_index <= int(e["end"])):
                continue
            active.append(e["type"])
            if e["type"] == "bs_port_fault":
                for p in e.get("ports", []):
                    ports[int(p)] = 0.0
            elif e["type"] == "ut_blockage":
                att[int(e["ut"])] += float(e.get("attenuation_db", 20.0))
            elif e["type"] == "interference_burst":
                sinr_loss += float(e.get("sinr_loss_db", 6.0))
        return {"blocked": blocked.tolist(), "attenuation_db": att.tolist(), "ports": ports.tolist(),
                "sinr_loss_db": sinr_loss, "events": active}

    def channel_hook(self, state: dict[str, Any]) -> Callable[[Any], Any]:
        import torch

        amp_ut = 10.0 ** (-(np.asarray(state["attenuation_db"]) + state["sinr_loss_db"]) / 20.0)
        ports = np.asarray(state["ports"])

        def hook(h_freq):
            # h_freq: [batch, rx, rx_ant, tx, tx_ant, ofdm_sym, fft]
            scale = torch.as_tensor(amp_ut, dtype=torch.float32)[None, None, None, :, None, None, None]
            mask = torch.as_tensor(ports, dtype=torch.float32)[None, None, :, None, None, None, None]
            return h_freq * (scale * mask).to(h_freq.dtype)

        return hook

    def cap_power(self, directives):
        """Clip every terminal's power to its class cap (body-worn exposure limit)."""
        if directives is None:
            return directives
        if directives.per_ut_power is not None:
            power = np.asarray(directives.per_ut_power, dtype=float)
        elif directives.active_ut_mask is not None:
            power = np.asarray(directives.active_ut_mask, dtype=float)
        else:
            power = np.ones(self.num_ut)
        capped = np.minimum(power, self.power_cap)
        if np.array_equal(capped, power) and directives.per_ut_power is not None:
            return directives
        return dataclasses.replace(directives, per_ut_power=capped.tolist())


def make_scenario(config: Any) -> IndustryScenario | None:
    """The run's scenario layer, or None when the config has no ``industry`` section."""
    if getattr(config, "industry", None) is None:
        return None
    system = config.system_runtime_config
    return IndustryScenario(config.industry, num_ut=int(system.get("num_ut", 4)),
                            num_bs_ant=int(system.get("num_bs_ant", 8)), seed=int(config.simulation.seed))
