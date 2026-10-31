"""
tests/chat_template_render.py — offline chat-template renderer for tests.

A faithful reproduction of llama-cpp-python 0.3.33's Jinja2ChatFormatter render
environment (ImmutableSandboxedEnvironment + the IgnoreGenerationTags /
loopcontrols extensions + the tojson/raise_exception/strftime_now globals). This
lets the cross-family templating assertion run anywhere — no llama_cpp install,
no multi-GB GGUFs — against the vendored template fixtures.

NOT authoritative for runtime behaviour: the real model + real llama-cpp is the
source of truth, covered by the hardware smoke tests (smoke_stream_ui.py,
smoke_real_gguf.py). This renderer exists to regression-guard the ASSEMBLER and
template STRUCTURE offline, and is kept in step with the real env via
scripts/manage_template_fixtures.py check.
"""
from __future__ import annotations

import json
from datetime import datetime

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class _IgnoreGenerationTags(Extension):
    """Mirror llama-cpp's handling of {% generation %}…{% endgeneration %}."""
    tags = {"generation"}

    def parse(self, parser):
        parser.stream.skip(1)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def _tojson(x, ensure_ascii=False, **kwargs):
    return json.dumps(x, ensure_ascii=ensure_ascii, **kwargs)


def render_chat(template: str, messages: list, *, eos_token: str = "<eos>",
                bos_token: str = "<bos>", add_generation_prompt: bool = True) -> str:
    """Render `messages` through a chat `template`, returning the prompt string."""
    env = ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True,
        extensions=[_IgnoreGenerationTags, jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = _tojson
    compiled = env.from_string(template)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    return compiled.render(
        messages=messages,
        eos_token=eos_token,
        bos_token=bos_token,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception,
        strftime_now=lambda fmt: datetime.now().strftime(fmt),
        functions=None, function_call=None, tools=None, tool_choice=None,
    )
