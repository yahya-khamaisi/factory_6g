from __future__ import annotations

from factory6g.models.resource_manager import RoundRobinResourceManager
from factory6g.sim.stages.common import jain_fairness_index


CONFIG = {"num_ut": 4}


def _replay(schedule: list[int], ebno_range: list[float]) -> list[list[int]]:
    """Mimic the stage loop: one call per open point per batch, points close early."""
    manager = RoundRobinResourceManager(num_active=2)
    counts = [[0] * CONFIG["num_ut"] for _ in ebno_range]
    for batch in range(max(schedule)):
        for i, ebno in enumerate(ebno_range):
            if batch >= schedule[i]:
                continue
            mask = manager.get_runtime_directives(CONFIG, ebno).active_ut_mask
            counts[i] = [c + m for c, m in zip(counts[i], mask)]
    return counts


def test_round_robin_split_is_independent_of_other_points_closing():
    # Run B seed 42 Rayleigh schedule: 11 points, five close at 100, one at 105,
    # six stay open to 200 (an even number => shared pointer froze each pair).
    ebno = [float(3 * i) for i in range(11)]
    schedule = [100, 100, 100, 100, 105] + [200] * 6
    counts = _replay(schedule, ebno)
    for i, per_ut in enumerate(counts):
        assert max(per_ut) - min(per_ut) <= 2, (ebno[i], per_ut)
        assert jain_fairness_index(per_ut) > 0.999


def test_round_robin_single_point_alternates_pairs():
    manager = RoundRobinResourceManager(num_active=2)
    masks = [manager.get_runtime_directives(CONFIG, 15.0).active_ut_mask for _ in range(4)]
    assert masks == [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]


def test_round_robin_rotates_over_distinct_ebno_values():
    # Dataset generation draws one Eb/N0 per record; rotation must survive that.
    manager = RoundRobinResourceManager(num_active=2)
    masks = [manager.get_runtime_directives(CONFIG, 0.1 * k).active_ut_mask for k in range(4)]
    assert masks == [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
