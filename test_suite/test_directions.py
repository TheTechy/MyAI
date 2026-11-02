"""
test_suite/test_directions.py
=============================
Unit tests for the route map DirectionsSkill delivers. No network: geocoding,
routing and the polite Nominatim delay are replaced with stand-ins.
"""
import base64
from pathlib import Path

import pytest

import myai_skills.directions as directions
from myai_skills.directions import DirectionsSkill


@pytest.fixture
def delivered(monkeypatch, tmp_path):
    """Run a directions request and split the SKILL_DELIVER payload into parts."""
    import core.file_generation as file_generation
    monkeypatch.setattr(file_generation, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(directions.time, "sleep", lambda s: None)
    places = {"Leeds": (53.80, -1.55, "Leeds"), "York": (53.96, -1.08, "York")}
    monkeypatch.setattr(directions, "_geocode", lambda place: places[place])
    monkeypatch.setattr(directions, "_get_route", lambda *a: {
        "distance": 40000, "duration": 2700, "legs": [{"steps": []}],
        "geometry": {"coordinates": [[-1.55, 53.80], [-1.08, 53.96]]},
    })
    payload = DirectionsSkill().execute("directions from Leeds to York")
    header, _description = payload.split("\n---\n", 1)
    tag, filename, mime, b64, disk_path = header.split("|", 4)
    assert tag == "SKILL_DELIVER"
    return {"filename": filename, "mime": mime, "disk_path": disk_path,
            "html": base64.b64decode(b64).decode("utf-8"), "out_dir": tmp_path}


def test_map_is_saved_so_it_survives_a_reload(delivered):
    saved = Path(delivered["disk_path"])
    assert saved.parent == delivered["out_dir"]
    assert saved.read_text(encoding="utf-8") == delivered["html"]
    assert delivered["mime"] == "text/html"


def test_map_uses_openstreetmaps_current_tile_address(delivered):
    # OSM retired the a./b./c. tile subdomains in favour of a single host.
    assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in delivered["html"]
    assert "{s}.tile.openstreetmap.org" not in delivered["html"]
