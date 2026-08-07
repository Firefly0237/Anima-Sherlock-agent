"""Shared fixtures for public package tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from anima.persona.contracts.pack import PACK_DATA_FILES
from tests.support.builders.pack_builder import write_pack

assert set(PACK_DATA_FILES) == {
    "profile.json",
    "lore.jsonl",
    "timeline.jsonl",
    "relationships.jsonl",
    "style_examples.jsonl",
    "safety.json",
}


@pytest.fixture()
def pack_dir(tmp_path: Path) -> Path:
    return write_pack(tmp_path)
