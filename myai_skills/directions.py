"""
myai_skills/directions.py
=========================
Driving directions skill — no API key required.

Stack
-----
  Geocoding : Nominatim  (nominatim.openstreetmap.org)
  Routing   : OSRM       (router.project-osrm.org)
  Map       : Leaflet.js (rendered in a self-contained HTML file)

Flow
----
  1. Parse origin and destination from the user query
  2. Geocode both addresses via Nominatim
  3. Fetch route (steps + full geometry) from OSRM
  4. Build a self-contained Leaflet HTML map file
  5. Return structured plain-text context for the LLM
     plus a [FILE:] block containing the map HTML

The LLM uses the plain-text context to compose a conversational
response (distance, duration, key steps). The [FILE:] block is
handled by file_generation.py and delivered to the user as a
download card — the HTML opens in the browser as an interactive map.

Nominatim fair-use policy
--------------------------
  - Max 1 request/second
  - Must include a descriptive User-Agent header
  - No bulk geocoding — single look-ups only (fine for this use case)
  Reference: https://operations.osmfoundation.org/policies/nominatim/
"""

from __future__ import annotations

import re
import time
import html
from typing import Optional

import requests

from .base import BaseSkill

# ── Constants ──────────────────────────────────────────────────────────────────

_USER_AGENT = "MyAI-DirectionsSkill/1.0"
_NOMINATIM  = "https://nominatim.openstreetmap.org/search"
_OSRM       = "https://router.project-osrm.org/route/v1/driving"
_TIMEOUT    = 10  # seconds

# Manoeuvre type → human-readable verb
_MANOEUVRE_VERBS: dict[str, str] = {
    "turn-right":           "Turn right",
    "turn-left":            "Turn left",
    "turn-slight-right":    "Bear right",
    "turn-slight-left":     "Bear left",
    "turn-sharp-right":     "Turn sharp right",
    "turn-sharp-left":      "Turn sharp left",
    "continue":             "Continue",
    "new-name":             "Continue onto",
    "depart":               "Head",
    "arrive":               "Arrive at your destination",
    "merge":                "Merge",
    "ramp":                 "Take the ramp",
    "on-ramp":              "Take the on-ramp",
    "off-ramp":             "Take the off-ramp",
    "fork":                 "Keep",
    "end-of-road":          "At the end of the road, turn",
    "roundabout":           "At the roundabout, take exit",
    "rotary":               "At the roundabout, take exit",
    "roundabout-turn":      "At the roundabout, turn",
    "exit-roundabout":      "Leave the roundabout",
    "exit-rotary":          "Leave the roundabout",
    "notification":         "Continue",
    "use-lane":             "Use lane",
}

# ── Regex patterns to extract origin / destination from query ──────────────────

_DIRECTIONS_PATTERNS = [
    # "directions/route/how do I get from X to Y"
    re.compile(
        r'(?:directions?|route|navigate|get|drive|travel|go)\s+'
        r'(?:from\s+)?(.+?)\s+to\s+(.+)',
        re.IGNORECASE,
    ),
    # "from X to Y"
    re.compile(
        r'from\s+(.+?)\s+to\s+(.+)',
        re.IGNORECASE,
    ),
    # "X to Y" (bare)
    re.compile(
        r'^(.+?)\s+to\s+(.+)$',
        re.IGNORECASE,
    ),
]

# Trailing noise to strip from parsed place names
_PLACE_NOISE = re.compile(
    r'\s*[\?\.\!]+$|'
    r'\s+(please|thanks|thank\s+you|driving|by\s+car|by\s+road)$',
    re.IGNORECASE,
)

# Personal-place words → the keyword we look for in the user's memories.
# Lets "directions from home to X" resolve to a remembered home address
# instead of geocoding the literal word "home".
_PERSONAL_PLACES: dict[str, str] = {
    "home":            "home",
    "my home":         "home",
    "home address":    "home",
    "my home address": "home",
    "my house":        "home",
    "house":           "home",
    "my place":        "home",
    "work":            "work",
    "my work":         "work",
    "office":          "work",
    "my office":       "work",
    "the office":      "work",
    "my workplace":    "work",
    "workplace":       "work",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_query(query: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (origin, destination) from a natural-language query."""
    for pattern in _DIRECTIONS_PATTERNS:
        m = pattern.search(query)
        if m:
            origin = _PLACE_NOISE.sub("", m.group(1).strip())
            dest   = _PLACE_NOISE.sub("", m.group(2).strip())
            if origin and dest:
                return origin, dest
    return None, None


def _extract_address(content: str) -> str:
    """
    Pull the address portion out of a free-text memory such as
    "my home address is 9 Wood Hill, Rothwell, West Yorkshire. LS26 0UN".
    Falls back to the whole string if no "… is/are/: …" lead is present.
    """
    m = re.search(r'\b(?:is|are|:)\s+(.+)$', content, re.IGNORECASE)
    return (m.group(1) if m else content).strip().rstrip(".")


def _resolve_personal_place(place: str, memories: Optional[list[dict]]) -> str:
    """
    If *place* is a personal reference ("home", "work", …) and the user has a
    memory describing that location, return the remembered address. Otherwise
    return *place* unchanged.
    """
    keyword = _PERSONAL_PLACES.get(place.strip().lower())
    if not keyword or not memories:
        return place
    for m in memories:
        content = m.get("content", "")
        if keyword in content.lower():
            return _extract_address(content)
    return place


def _geocode(place: str) -> tuple[float, float, str]:
    """
    Resolve a place name to (lat, lon, display_name).
    Raises ValueError if no result found.
    """
    r = requests.get(
        _NOMINATIM,
        params={
            "q":              place,
            "format":         "json",
            "limit":          1,
            "addressdetails": 1,
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        raise ValueError(f"Could not find location: '{place}'")
    best = results[0]
    return float(best["lat"]), float(best["lon"]), best.get("display_name", place)


def _get_route(
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
) -> dict:
    """
    Fetch driving route from OSRM.
    Returns the raw route dict from the API.
    Raises ValueError on routing failure.
    """
    url = f"{_OSRM}/{o_lon},{o_lat};{d_lon},{d_lat}"
    r   = requests.get(
        url,
        params={
            "steps":     "true",
            "overview":  "full",
            "geometries":"geojson",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError("OSRM returned no route.")
    return data["routes"][0]


def _format_distance(metres: float) -> str:
    if metres >= 1000:
        return f"{metres / 1000:.1f} km"
    return f"{int(metres)} m"


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    return f"{h} hr {m} min" if m else f"{h} hr"


def _step_instruction(step: dict) -> str:
    """Build a human-readable instruction from an OSRM step."""
    manoeuvre = step.get("maneuver", {})
    m_type    = manoeuvre.get("type", "")
    m_mod     = manoeuvre.get("modifier", "")
    name      = step.get("name", "")

    key  = f"{m_type}-{m_mod}" if m_mod else m_type
    verb = _MANOEUVRE_VERBS.get(key) or _MANOEUVRE_VERBS.get(m_type, "Continue")

    if m_type == "arrive":
        return "Arrive at your destination"
    if m_type == "depart":
        direction = f" {m_mod}" if m_mod else ""
        road      = f" on {name}" if name else ""
        return f"Head{direction}{road}"
    if m_type in ("roundabout", "rotary"):
        exit_num = manoeuvre.get("exit", "")
        suffix   = f" {exit_num}" if exit_num else ""
        road     = f" onto {name}" if name else ""
        return f"{verb}{suffix}{road}"

    road = f" onto {name}" if name else ""
    return f"{verb}{road}"


def _build_plain_text_context(
    origin_display: str,
    dest_display: str,
    distance_m: float,
    duration_s: float,
    steps: list[dict],
) -> str:
    """Build the plain-text context string for the LLM."""
    lines = [
        f"Origin:      {origin_display}",
        f"Destination: {dest_display}",
        f"Distance:    {_format_distance(distance_m)}",
        f"Duration:    {_format_duration(duration_s)}",
        "",
        "Turn-by-turn directions:",
    ]
    for i, step in enumerate(steps, 1):
        instr = _step_instruction(step)
        dist  = _format_distance(step.get("distance", 0))
        lines.append(f"  {i:2d}. {instr}  ({dist})")
    return "\n".join(lines)


def _build_map_html(
    origin: str,
    destination: str,
    origin_display: str,
    dest_display: str,
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
    distance_m: float,
    duration_s: float,
    steps: list[dict],
    geometry_coords: list,
) -> str:
    """
    Build a fully self-contained Leaflet HTML map file.
    All coordinates are embedded as JSON literals — no external data calls.
    """

    # Build steps HTML for the sidebar
    steps_html_parts = []
    for step in steps:
        instr    = _step_instruction(step)
        dist_str = _format_distance(step.get("distance", 0))
        m_type   = step.get("maneuver", {}).get("type", "")

        if m_type == "depart":
            icon = "&#9654;"   # ▶ start
        elif m_type == "arrive":
            icon = "&#9679;"   # ● finish
        elif "right" in step.get("maneuver", {}).get("modifier", ""):
            icon = "&#8594;"   # → turn right
        elif "left" in step.get("maneuver", {}).get("modifier", ""):
            icon = "&#8592;"   # ← turn left
        else:
            icon = "&#8593;"   # ↑ straight on

        steps_html_parts.append(
            f'<li><span class="icon">{icon}</span>'
            f'<span class="instr">{html.escape(instr)}</span>'
            f'<span class="dist">{html.escape(dist_str)}</span></li>'
        )
    steps_html = "\n".join(steps_html_parts)

    # Leaflet coordinate array — [lat, lon] pairs
    coords_js = str([[lat, lon] for lon, lat in geometry_coords])

    dist_str = _format_distance(distance_m)
    dur_str  = _format_duration(duration_s)

    origin_safe = html.escape(origin)
    dest_safe   = html.escape(destination)
    o_disp      = html.escape(origin_display.split(",")[0])
    d_disp      = html.escape(dest_display.split(",")[0])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{origin_safe} → {dest_safe}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ display: flex; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; color: #1a1a1a; background: #f5f5f5; }}
  #map {{ flex: 1; }}
  #panel {{ width: 300px; display: flex; flex-direction: column; background: #fff; border-left: 1px solid #e0e0e0; overflow: hidden; }}
  #header {{ padding: 16px; border-bottom: 1px solid #e8e8e8; background: #fff; }}
  #header h1 {{ font-size: 14px; font-weight: 600; color: #111; line-height: 1.4; margin-bottom: 6px; }}
  #meta {{ display: flex; gap: 12px; }}
  .badge {{ font-size: 12px; color: #555; background: #f0f0f0; padding: 3px 8px; border-radius: 12px; }}
  #steps {{ flex: 1; overflow-y: auto; padding: 8px 0; }}
  #steps ul {{ list-style: none; }}
  #steps li {{ display: flex; align-items: flex-start; gap: 10px; padding: 9px 16px; border-bottom: 1px solid #f0f0f0; }}
  #steps li:last-child {{ border-bottom: none; }}
  .icon {{ font-size: 16px; min-width: 20px; text-align: center; margin-top: 1px; color: #555; }}
  .instr {{ flex: 1; font-size: 13px; line-height: 1.4; color: #222; }}
  .dist  {{ font-size: 11px; color: #888; white-space: nowrap; padding-top: 2px; }}
  #footer {{ padding: 10px 16px; border-top: 1px solid #e8e8e8; font-size: 11px; color: #aaa; text-align: center; }}
  .leaflet-container {{ font-family: inherit; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <div id="header">
    <h1>{o_disp} &rarr; {d_disp}</h1>
    <div id="meta">
      <span class="badge">&#128694; {dist_str}</span>
      <span class="badge">&#128336; {dur_str}</span>
    </div>
  </div>
  <div id="steps">
    <ul>
{steps_html}
    </ul>
  </div>
  <div id="footer">Map data &copy; OpenStreetMap contributors</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  var coords = {coords_js};
  var map = L.map('map');
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }}).addTo(map);

  var route = L.polyline(coords, {{ color: '#1a73e8', weight: 5, opacity: 0.85 }}).addTo(map);
  map.fitBounds(route.getBounds(), {{ padding: [40, 40] }});

  var greenIcon = L.divIcon({{
    html: '<div style="background:#1D9E75;color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.3)">A</div>',
    iconSize: [28, 28], iconAnchor: [14, 14], className: ''
  }});
  var redIcon = L.divIcon({{
    html: '<div style="background:#D85A30;color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.3)">B</div>',
    iconSize: [28, 28], iconAnchor: [14, 14], className: ''
  }});

  L.marker(coords[0], {{icon: greenIcon}}).bindPopup('<b>{o_disp}</b>').addTo(map).openPopup();
  L.marker(coords[coords.length - 1], {{icon: redIcon}}).bindPopup('<b>{d_disp}</b>').addTo(map);
</script>
</body>
</html>"""


# ── Skill class ────────────────────────────────────────────────────────────────

class DirectionsSkill(BaseSkill):
    """
    Driving directions skill.

    Returns a plain-text context string (distance, duration, turn-by-turn)
    for the LLM to compose a conversational response, plus a [FILE:] block
    containing a self-contained Leaflet HTML map the user can open in their
    browser.
    """

    name        = "directions"
    description = "Driving directions with interactive map (OSM + OSRM, no API key)"

    def execute(self, query: str, memories: Optional[list[dict]] = None) -> str:
        origin, destination = _parse_query(query)

        if not origin or not destination:
            return (
                "Could not parse origin and destination from the query. "
                "Please specify both, e.g. 'Directions from Rochdale to Manchester'."
            )

        # Resolve personal references ("home", "work") against the user's
        # remembered addresses before geocoding.
        origin      = _resolve_personal_place(origin, memories)
        destination = _resolve_personal_place(destination, memories)

        try:
            # Small polite delay between Nominatim calls (fair-use policy)
            o_lat, o_lon, o_display = _geocode(origin)
            time.sleep(0.5)
            d_lat, d_lon, d_display = _geocode(destination)
        except ValueError as e:
            return f"Geocoding error: {e}"
        except requests.RequestException as e:
            return f"Network error during geocoding: {e}"

        try:
            route = _get_route(o_lat, o_lon, d_lat, d_lon)
        except (ValueError, requests.RequestException) as e:
            return f"Routing error: {e}"

        distance_m = route.get("distance", 0)
        duration_s = route.get("duration", 0)
        legs       = route.get("legs", [])
        steps      = []
        for leg in legs:
            steps.extend(leg.get("steps", []))

        geometry_coords = route.get("geometry", {}).get("coordinates", [])

        plain_text = _build_plain_text_context(
            o_display, d_display, distance_m, duration_s, steps
        )

        map_html = _build_map_html(
            origin, destination,
            o_display, d_display,
            o_lat, o_lon,
            d_lat, d_lon,
            distance_m, duration_s,
            steps, geometry_coords,
        )

        # Sanitise filename
        def _slug(s: str) -> str:
            return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')[:20]

        filename = f"route_{_slug(origin)}_to_{_slug(destination)}.html"

        import base64
        b64data = base64.b64encode(map_html.encode("utf-8")).decode("ascii")

        # SKILL_DELIVER: file goes straight to the SSE pipeline,
        # plain_text goes to the LLM for its conversational response.
        # This avoids putting the HTML into the LLM context window.
        return f"SKILL_DELIVER|{filename}|text/html|{b64data}\n---\n{plain_text}"