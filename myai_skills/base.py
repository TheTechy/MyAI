"""
skills/base.py
==============
Abstract base class that all MyAI skills must inherit from.

A skill is a self-contained module that:
  - Executes against user queries and returns plain text context
    that the LLM uses to compose its final answer

Routing is handled by the LLM intent classifier in llm_inference.py.
Skills no longer need to implement routing logic.

Adding a new skill
------------------
1. Create skills/my_skill.py inheriting from BaseSkill
2. Set name and description
3. Implement execute()
4. Register it in skills/__init__.py
5. Add the skill name to the classifier prompt in llm_inference.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """
    Abstract base for all MyAI skills.

    Subclasses must define:
        name        : str  — short identifier e.g. "web_search"
        description : str  — one-line human description

    Subclasses must implement:
        execute(query)     — perform the skill, return plain text
    """

    name:        str = "base"
    description: str = "Base skill"

    @abstractmethod
    def execute(self, query: str) -> str:
        """
        Run the skill against *query* and return plain text context
        for the LLM to use when composing its answer.

        Should never raise — catch exceptions internally and return
        a descriptive error string so the LLM can report gracefully.
        """
        ...

    def __repr__(self) -> str:
        return f"<Skill: {self.name}>"
