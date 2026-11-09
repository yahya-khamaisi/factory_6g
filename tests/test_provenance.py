from __future__ import annotations

import json

from factory6g.sim.config import load_config
from factory6g.sim.provenance import build_provenance, canonical_json_sha256, source_sha256, write_provenance

from .conftest import make_tiny_config, write_config


def test_config_hash_ignores_key_order():
    assert canonical_json_sha256({"a": 1, "b": [1, 2]}) == canonical_json_sha256({"b": [1, 2], "a": 1})
    assert canonical_json_sha256({"a": 1}) != canonical_json_sha256({"a": 2})


def test_provenance_record_and_resume_files(tmp_path):
    record = build_provenance(raw_config={"simulation": {"seed": 7}}, config_path="c.json", seed=7, deterministic=True, argv=["run"])
    assert record["source_sha256"] == source_sha256()
    assert record["seed"] == 7 and record["deterministic_ops"] is True
    assert "numpy" in record["versions"]
    first = write_provenance(tmp_path, record)
    second = write_provenance(tmp_path, record)
    assert first.name == "provenance.json"
    assert second.name == "provenance_resume_1.json"
    assert json.loads(first.read_text())["config_sha256"] == record["config_sha256"]


def test_deterministic_flag_is_optional_and_parsed(tmp_path):
    data = make_tiny_config(str(tmp_path / "results"))
    assert load_config(write_config(tmp_path, data)).simulation.deterministic is False
    data["simulation"]["deterministic"] = True
    assert load_config(write_config(tmp_path, data)).simulation.deterministic is True
