#!/usr/bin/env python3
"""
tests/smoke_cross_family_templating.py — F2 S1 T4

Standing regression guard: identical assembled messages must produce a
correctly-STRUCTURED prompt across every model family. Renders the ASSEMBLER's
output (core.inference.messages) through each vendored template fixture and
checks family-agnostic invariants. Catches assembler regressions (lost
system-first ordering, broken role mapping, dropped turns) against real templates
without needing llama_cpp or the GGUFs.

Fixtures are auto-discovered from tests/fixtures/chat_templates/*.jinja — add a
model by dropping a file there, remove one by deleting its file. No code change.
Refresh/verify fixtures with scripts/manage_template_fixtures.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── bootstrap: find the repo root (dir containing core/inference) + this dir ────
_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "core" / "inference").is_dir():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(_here.parent))

from chat_template_render import render_chat          # noqa: E402
from core.inference.messages import assemble_messages  # noqa: E402

FIXTURES = _here.parent / "fixtures" / "chat_templates"

# Distinct sentinels so each turn is unambiguously locatable in the rendered text.
SYS, U1, A1, U2 = "SYSMARKER", "USERONE", "ASSISTANTONE", "USERTWO"


def main() -> int:
    fixtures = sorted(FIXTURES.glob("*.jinja"))
    if not fixtures:
        print(f"FAIL no template fixtures in {FIXTURES} — a standing test with no "
              f"fixtures is a false green")
        return 1
    print(f"discovered {len(fixtures)} template fixture(s)")

    # Identical assembled messages for every family (the T4 'identical messages').
    messages = assemble_messages(
        SYS,
        [{"role": "user", "content": U1}, {"role": "model", "content": A1}],
        U2,
        supports_system_role=True,
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]

    failures = []

    def check(ok, label):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    for fx in fixtures:
        name = fx.stem
        print(f"\n─── {name} ───")
        template = fx.read_text(encoding="utf-8")
        try:
            out = render_chat(template, messages)
        except Exception as e:                       # noqa: BLE001
            check(False, f"renders without error ({type(e).__name__}: {e})")
            continue

        # 1) Every turn's content survives into the prompt.
        positions = {tag: out.find(tag) for tag in (SYS, U1, A1, U2)}
        check(all(p != -1 for p in positions.values()),
              "all four turns present in the rendered prompt")
        if any(p == -1 for p in positions.values()):
            continue

        # 2) Turns appear in the SAME ORDER as the messages array.
        check(positions[SYS] < positions[U1] < positions[A1] < positions[U2],
              "turn order preserved (system → user → assistant → user)")

        # 3) System is FIRST (the invariant both families enforce), and appears
        #    exactly once — guards against a silent merge into every turn.
        check(positions[SYS] == min(positions.values()), "system turn is first")
        check(out.count(SYS) == 1, "system content appears exactly once")

        # 4) The template actually added structure (markers), not just concatenated.
        content_len = len(SYS) + len(U1) + len(A1) + len(U2)
        check(len(out) > content_len, "template added turn markers around content")

    print("\n" + "=" * 56)
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print(f"ALL GREEN across {len(fixtures)} families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
