"""Text normalisation: verbatim quote containment (M2) and topic matching (M3)."""
from __future__ import annotations

import re
import string
import unicodedata

_PUNCT = string.punctuation + "“”‘’«»…–—"
STOP_WORDS = frozenset("""
a an and are as at be by for from in into is it its of on or that the their this to with within without via
using use based over under between than then these those which what when where how why who whom whose
""".split())


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("\\(", " ").replace("\\)", " ").replace("$", " ")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalise_quote(q: str) -> str:
    return normalise(q).strip(_PUNCT + " ")


def quote_in(quote: str, answer: str) -> bool:
    """A pass counts only when its quote (normalised) is a substring of the normalised answer."""
    nq = normalise_quote(quote)
    return bool(nq) and nq in normalise(answer)


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", normalise(s))}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def content_words(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalise(s)) if len(t) >= 4 and t not in STOP_WORDS]


def topic_named_in(topic: str, text: str) -> bool:
    """True when the topic, or every content word of it, appears in the text (both normalised)."""
    nt, nx = normalise(topic), normalise(text)
    if nt and nt in nx:
        return True
    words = content_words(topic)
    return bool(words) and all(w in nx for w in words)
