"""Run viewer: exports a run's per-batch trace and serves a browser viewer for it.

Runs in the TensorFlow-free ``viz`` image (``Dockerfile.viz``). Nothing here
imports TensorFlow or the Sionna PHY, so it only needs numpy and, for the
optional radio map, Sionna RT.
"""
