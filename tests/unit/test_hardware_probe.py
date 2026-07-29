from __future__ import annotations

import json
from pathlib import Path

from edgeforge_core.system.hardware_probe import (
    collect_system_snapshot,
    save_system_snapshot,
)


def test_system_snapshot_contains_expected_sections() -> None:
    """The snapshot should contain all major information groups."""
    snapshot = collect_system_snapshot()

    expected_sections = {
        "generated_at_utc",
        "system",
        "python",
        "packages",
        "nvidia",
        "pytorch",
        "onnx_runtime",
        "openvino",
        "git",
    }

    assert expected_sections.issubset(snapshot.keys())


def test_system_snapshot_contains_valid_memory() -> None:
    """The detected system should have a positive RAM value."""
    snapshot = collect_system_snapshot()

    ram_gb = snapshot["system"]["ram_gb"]

    assert isinstance(ram_gb, float)
    assert ram_gb > 0


def test_python_version_is_recorded() -> None:
    """The running Python version should be stored."""
    snapshot = collect_system_snapshot()

    python_version = snapshot["python"]["version"]

    assert isinstance(python_version, str)
    assert python_version.startswith("3.12")


def test_snapshot_can_be_saved_as_json(tmp_path: Path) -> None:
    """A snapshot should save as valid readable JSON."""
    snapshot = collect_system_snapshot()
    output_path = tmp_path / "system_info.json"

    saved_path = save_system_snapshot(snapshot, output_path)

    assert saved_path.exists()

    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_data["system"]["ram_gb"] > 0
    assert "pytorch" in saved_data
