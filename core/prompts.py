# prompts.py
import json
import textwrap
from datetime import date
from pathlib import Path

from core.db import list_memories, get_user_by_id

# ── Date variables ─────────────────────────────────────────────────────────────
_today = date.today()
current_date  = _today.strftime("%d-%m-%Y")
current_day   = str(_today.day)             # e.g. "9" (cross platform)
current_month = _today.strftime("%B")       # e.g. "June"
current_year  = _today.strftime("%Y")       # e.g. "2026"


def _ordinal(n: int) -> str:
    """Return "9th", "1st", "22nd" etc."""
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

current_day_ordinal = _ordinal(_today.day)  # e.g. "9th"

# ── Age band personalities ────────────────────────────────────────────────────
# Loaded from data/age_band_prompts.json at import time. Only the personality
# prefix is stored there — the rules block below applies uniformly to every
# user regardless of age band, so search / formatting / uncertainty handling
# stay consistent.
_AGE_BAND_JSON_PATH = Path(__file__).parent.parent / "data" / "age_band_prompts.json"


def _load_age_band_prompts() -> dict:
    """
    Load the age-band personality prefixes from the JSON file.
    Silently falls back to an empty dict if the file is missing or malformed —
    in that case every user gets the same behaviour via _FALLBACK_PERSONALITY.
    """
    try:
        with _AGE_BAND_JSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop the _comment key, keep only real bands
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"[prompts] Could not load age band JSON ({e}); falling back to defaults.")
        return {}


_AGE_BAND_PERSONALITIES = _load_age_band_prompts()

# Used when the age band isn't in the JSON (missing file, unknown band, etc.).
# Matches the adult personality exactly so behaviour is unchanged.
_FALLBACK_PERSONALITY = "You are MyAI, a knowledgeable and helpful assistant. You respond naturally and directly."


# ── Shared date preamble ──────────────────────────────────────────────────────
# Model-agnostic and factored out of the per-band JSON (it used to be duplicated
# in every band). Anchors relative dates to today. Deliberately states NO training
# cutoff: v2 swaps models freely, so a hard cutoff would be wrong for most models —
# recency is handled by routing/search, not by a date in the prompt.
def _date_preamble() -> str:
    return (
        f"Today is {current_day_ordinal} {current_month} {current_year} "
        f"({current_date}). When the user says 'this year' they mean "
        f"{current_year}; when they say 'this month' they mean "
        f"{current_month} {current_year}."
    )


# ── Shared rules block ────────────────────────────────────────────────────────
# The rules apply to every user regardless of age band. Editing these has
# big consequences (e.g. SEARCH behaviour, response formatting) so they live
# in code where a syntax error is loud and immediate.
_PROMPT_RULES = textwrap.dedent(f"""\
---

**How to handle knowledge & uncertainty**
- Answer directly from your own knowledge when you can do so confidently.
- If you are partially unsure, share what you know and flag uncertainty clearly — distinguish between "I don't know" and "I'm not certain but...". Never fabricate facts.
- For genuinely complex or contested topics, acknowledge that — give a useful answer without false confidence.

---

**How to respond**
Aim to be genuinely helpful, not just technically correct. That means:
- Lead with the answer, then explain or elaborate as needed. Don't restate the question.
- Match depth to the question. A quick factual query warrants a short answer; a nuanced question deserves a fuller one. Don't pad, but don't truncate when detail adds value.
- Think through problems carefully before responding, especially for multi-step reasoning, code, or analysis. It's fine to work through something step by step rather than jump to a conclusion.
- For tasks like writing, coding, or analysis, just do the task — don't preface it with unnecessary commentary about what you're about to do.
- When a question is ambiguous, make a reasonable interpretation and state your assumption, rather than asking for clarification before attempting an answer.

---
**Formatting**
- Use markdown only when it genuinely aids clarity — code blocks for code, lists when items are truly enumerable, headers only for longer structured responses.
- Prefer prose over bullet points for explanations and reasoning. Bullets fragment ideas that flow better as sentences.
- Don't use bold for decoration — reserve it for genuinely critical terms or warnings.
- Never use LaTeX or MathJax notation ($, $$, \\text{{}}, \\frac{{}} etc.) — the interface does not render it. Write mathematical and scientific expressions in plain text using Unicode characters where helpful. For example: write 1s² 2s² 2p⁶ not $$1s^2 2s^2 2p^6$$, and write CO₂ not $\\text{{CO}}_2$.
- Keep responses as long as they need to be, and no longer.
""")


def _get_personality(age_band: str) -> str:
    """
    Return the personality prefix for an age band, with date placeholders
    substituted in. Falls back to the adult-style default when the band is
    unknown or the JSON is missing.
    """
    template = _AGE_BAND_PERSONALITIES.get(age_band) or _FALLBACK_PERSONALITY
    try:
        return template.format(
            current_day_ordinal=current_day_ordinal,
            current_month=current_month,
            current_year=current_year,
            current_date=current_date,
        )
    except (KeyError, IndexError) as e:
        # A malformed placeholder in the JSON shouldn't take the app down —
        # fall back to the safe default and log so the operator can fix it.
        print(f"[prompts] Bad placeholder in age band '{age_band}': {e}. Using fallback.")
        return _FALLBACK_PERSONALITY


# ── Memory injection ──────────────────────────────────────────────────────────
def _memory_block(user_id: str) -> str:
    """
    Build a system-prompt section listing everything the user has explicitly
    asked to be remembered. Returns an empty string when there are none.
    """
    try:
        memories = list_memories(user_id)
    except Exception:
        return ""
    if not memories:
        return ""

    facts = "\n".join(f"- {m['content']}" for m in memories)
    return textwrap.dedent("""\

        ---

        **What you know about this user**

        The facts below are things the user has personally told you. Treat them
        as background knowledge — like things you already know about a friend —
        not as a list to read out.

        How to use them:

        - When a question relates to a fact, answer using ONLY the specific
          relevant fact, in a normal sentence. Do not introduce it as "I remember…"
          or "Based on what you've told me…".
        - Never list, quote, recite, or paraphrase the whole set of facts.
        - Never mention that you have "stored memories" or a "memory list".
        - If no fact is relevant, ignore them entirely and answer normally.
        - If asked directly "what do you remember about me", the system handles
          that separately — you will not see that question.

        Known facts:

        """) + facts + "\n"


# ── Public API ────────────────────────────────────────────────────────────────
def get_prompt_for_user(user_id: str) -> str:
    """
    Return the assembled system prompt for a user (UUID).
    Structure: [personality prefix based on age_band] + [rules] + [memories]

    Looks up the user's age_band in the DB. If the user doesn't exist or the
    band is unknown, falls back to the adult-style default.
    """
    user = None
    try:
        user = get_user_by_id(user_id)
    except Exception:
        pass  # DB error — fall through to fallback

    age_band = (user or {}).get("age_band") or "adult"
    personality = _get_personality(age_band)
    preamble    = _date_preamble()
    memories    = _memory_block(user_id)

    return f"{personality}\n\n{preamble}\n\n{_PROMPT_RULES}{memories}"
