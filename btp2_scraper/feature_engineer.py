"""
Feature engineer: computes Group C text features.

Designed to be CPU-cheap so it runs synchronously after the network-bound
extractors. Pure Python where reasonable; no GPU needed.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import config
from .schema import VideoRecord

logger = logging.getLogger(__name__)

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False
    logger.warning("textblob not installed; sentiment features will be None")


_RE_NUMBER = re.compile(r"\d")
_RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_RE_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)
_RE_WORD = re.compile(r"\b[A-Za-z]+\b")
_RE_SENTENCE = re.compile(r"[.!?]+")

_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]",
    flags=re.UNICODE,
)

_CTA_TERMS = {
    "subscribe", "sub", "like", "comment", "share", "follow", "join",
    "click", "watch", "hit the bell", "smash that like", "ring the bell",
    "check out", "let me know", "drop a comment", "tap subscribe",
}
_CTA_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _CTA_TERMS) + r")\b",
    flags=re.IGNORECASE,
)


def _count_syllables(word: str) -> int:
    word = word.lower()
    if not word:
        return 0
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    groups = _RE_VOWEL_GROUP.findall(word)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> Optional[float]:
    """Flesch Reading Ease. Higher = easier. 60–70 is plain English."""
    if not text or len(text) < 10:
        return None

    sentences = [s for s in _RE_SENTENCE.split(text) if s.strip()]
    words = _RE_WORD.findall(text)
    if not words or not sentences:
        return None

    syllables = sum(_count_syllables(w) for w in words)
    n_sent = max(1, len(sentences))
    n_words = len(words)

    score = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    return round(score, 2)


def _sentiment_pair(text: str) -> tuple[Optional[float], Optional[float]]:
    if not _HAS_TEXTBLOB or not text:
        return (None, None)
    try:
        blob = TextBlob(text)
        return (float(blob.sentiment.polarity), float(blob.sentiment.subjectivity))
    except Exception:
        return (None, None)


def _caps_ratio(text: str) -> Optional[float]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    upper = sum(1 for c in letters if c.isupper())
    return round(upper / len(letters), 4)


def engineer_one(rec: VideoRecord) -> None:
    """Mutate a record in place, filling all Group C features."""
    title = rec.title or ""
    desc = rec.description or ""
    combined = f"{title} {desc}".strip()

    rec.title_length_chars = len(title)
    rec.title_word_count = len(title.split())
    rec.title_has_number = bool(_RE_NUMBER.search(title)) if title else False
    rec.title_has_question = "?" in title
    rec.title_caps_ratio = _caps_ratio(title)
    rec.title_emoji_count = len(_RE_EMOJI.findall(title))

    title_pol, title_sub = _sentiment_pair(title)
    rec.title_sentiment_polarity = title_pol
    rec.title_subjectivity = title_sub

    rec.description_length_chars = len(desc)
    rec.description_word_count = len(desc.split())
    rec.description_url_count = len(_RE_URL.findall(desc))

    desc_pol, desc_sub = _sentiment_pair(desc)
    rec.description_sentiment_polarity = desc_pol
    rec.description_subjectivity = desc_sub

    pols = [p for p in (title_pol, desc_pol) if p is not None]
    rec.combined_sentiment = sum(pols) / len(pols) if pols else None

    rec.flesch_readability = flesch_reading_ease(desc)

    cta_matches = _CTA_PATTERN.findall(combined)
    rec.cta_word_count = len(cta_matches)
    rec.cta_presence = len(cta_matches) > 0


def engineer_batch(records: list[VideoRecord]) -> None:
    """Apply engineer_one to a batch in parallel."""
    if not records:
        return

    if _HAS_TEXTBLOB:
        try:
            _ = TextBlob("hello world").sentiment
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=config.FEATURE_WORKERS) as pool:
        list(pool.map(engineer_one, records))
