"""Provenance record written next to every run: what code, config and stack produced it.

``provenance.json`` holds the config hash, a hash of the package source, the git
commit when the checkout is mounted, the command line, library versions and
the seed. Two runs with equal ``config_sha256``, ``source_sha256`` and seed came
from the same inputs; the record is how a result file is traced back to them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/factory6g


def canonical_json_sha256(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_sha256(root: Path = PACKAGE_ROOT) -> str:
    """Hash of every Python file in the package, by relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_head(start: Path = PACKAGE_ROOT) -> dict[str, str] | None:
    """Commit and branch read from .git directly (the containers have no git binary)."""
    for parent in [start, *start.parents]:
        git_dir = parent / ".git"
        if not git_dir.is_dir():
            continue
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return {"commit": head, "branch": "(detached)"}
        ref = head[5:]
        ref_file = git_dir / ref
        if ref_file.exists():
            return {"commit": ref_file.read_text(encoding="utf-8").strip(), "branch": ref.removeprefix("refs/heads/")}
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return {"commit": line.split()[0], "branch": ref.removeprefix("refs/heads/")}
        return {"commit": "unknown", "branch": ref.removeprefix("refs/heads/")}
    return None


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "tensorflow", "sionna", "scipy", "mitsuba", "drjit"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except Exception:
                continue
        out[name] = str(getattr(module, "__version__", "unknown"))
    return out


def build_provenance(*, raw_config: dict[str, Any], config_path: str, seed: int, deterministic: bool, argv: list[str]) -> dict[str, Any]:
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config_path": config_path,
        "config_sha256": canonical_json_sha256(raw_config),
        "source_sha256": source_sha256(),
        "git": git_head(),
        "argv": list(argv),
        "seed": int(seed),
        "deterministic_ops": bool(deterministic),
        # Set FACTORY6G_IMAGE to the image digest (docker image inspect) to record it.
        "image": os.environ.get("FACTORY6G_IMAGE", "unrecorded"),
        "versions": _versions(),
    }


def write_provenance(run_dir: Path, record: dict[str, Any]) -> Path:
    """Write provenance.json; a resumed run adds provenance_resume_<n>.json instead of overwriting."""
    run_dir = Path(run_dir)
    path = run_dir / "provenance.json"
    n = 1
    while path.exists():
        path = run_dir / f"provenance_resume_{n}.json"
        n += 1
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
