"""
Helpers for parsing final-answer directives from LLM responses.
"""

from __future__ import annotations

import ast
import re
from typing import Optional


def extract_final_literal(text: str) -> Optional[str]:
    """
    Extract a literal answer from final()/FINAL() directives when the model
    returns a one-liner instead of a full executable block.
    """
    stripped = text.strip()
    if not stripped:
        return None

    match = re.fullmatch(r"(?is)(?:final|FINAL)\((.*)\)\s*", stripped)
    if not match:
        return None

    inner = match.group(1).strip()
    if not inner:
        return ""
    try:
        value = ast.literal_eval(inner)
    except Exception:
        return None
    return str(value)


def looks_like_final_var(text: str) -> bool:
    """Return True when the text looks like FINAL_VAR(...)."""
    return bool(re.fullmatch(r"(?is)FINAL_VAR\((.*)\)\s*", text.strip()))
