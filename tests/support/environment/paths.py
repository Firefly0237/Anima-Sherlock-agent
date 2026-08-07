"""Stable repository paths for tests that use checked-in assets."""

from __future__ import annotations

from pathlib import Path


def _find_project_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "anima").is_dir():
            return candidate
    raise RuntimeError(f"could not locate the project root from {source}")


PROJECT_ROOT = _find_project_root(Path(__file__))
TEST_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
