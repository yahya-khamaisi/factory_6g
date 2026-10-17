"""The Chapter 3 resource-grid figure must show the pilot layout Sionna actually uses."""

from __future__ import annotations

import numpy as np
import pytest

from factory6g.visualization.thesis_methodology_figures import (
    FFT_SIZE,
    NUM_OFDM_SYMBOLS,
    NUM_UT,
    PILOT_OFDM_SYMBOLS,
    kronecker_pilot_owner,
)


def test_kronecker_pilot_owner_matches_sionna_resource_grid() -> None:
    ofdm = pytest.importorskip("sionna.phy.ofdm")
    rg = ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=30e3,
        num_tx=NUM_UT,
        num_streams_per_tx=1,
        cyclic_prefix_length=20,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=list(PILOT_OFDM_SYMBOLS),
    )
    mask = np.asarray(rg.pilot_pattern.mask)[:, 0]  # [tx, symbol, subcarrier]
    pilots = np.asarray(rg.pilot_pattern.pilots)[:, 0]  # [tx, num_pilots]

    owner = kronecker_pilot_owner()
    for u in range(NUM_UT):
        # Pilot positions (mask) are the same full pilot symbols for every terminal.
        np.testing.assert_array_equal(mask[u] == 1, owner >= 0)
        # Terminal u's non-zero pilots are exactly the cells the figure gives it.
        nonzero = np.zeros_like(mask[u], dtype=bool)
        positions = np.argwhere(mask[u] == 1)
        for (sym, k), value in zip(positions, pilots[u]):
            nonzero[sym, k] = abs(value) > 0
        np.testing.assert_array_equal(nonzero, owner == u)
    # Data resource elements per terminal: 12 symbols x 128 subcarriers.
    assert int((owner < 0).sum()) == rg.num_data_symbols == 1536
