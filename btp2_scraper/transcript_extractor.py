"""
Transcript extractor.

Two-tier strategy (free fast path → expensive fallback):

  1. youtube-transcript-api: hits YouTube's caption servers directly,
     no API key needed, no quota. Works for ~85% of English videos.

  2. (Optional) Whisper: local ASR on yt-dlp-downloaded audio.
     Slow and GPU-bound. Only runs when explicitly enabled.

Output is persisted to disk as one JSON file per video so transcripts can
be loaded lazily by feature engineering / model training without bloating
the main parquet shards.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import config
from .schema import VideoRecord

logger = logging.getLogger(__name__)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _HAS_TRANSCRIPT_API = True
    # v1.x uses an instance; v0.x used class methods. Detect which is installed.
    _YTT_V1 = not hasattr(YouTubeTranscriptApi, "get_transcript")
except ImportError:
    _HAS_TRANSCRIPT_API = False
    _YTT_V1 = False
    logger.warning("youtube-transcript-api not installed; transcript extraction will be skipped")


def _transcript_path(video_id: str) -> Path:
    return config.TRANSCRIPTS_DIR / f"{video_id}.json"


def _save_transcript(video_id: str, source: str, segments: list[dict], language: str) -> None:
    path = _transcript_path(video_id)
    payload = {
        "video_id": video_id,
        "source": source,
        "language": language,
        "segments": segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))


def _flatten_text(segments: list[dict]) -> str:
    parts = [s.get("text", "").strip() for s in segments if s.get("text")]
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    return text.strip()


def _compute_speech_rate_wpm(segments: list[dict], full_text: str) -> Optional[float]:
    if not segments or not full_text:
        return None
    last = segments[-1]
    span_sec = (last.get("start", 0) + last.get("duration", 0))
    if span_sec <= 0:
        return None
    words = len(full_text.split())
    return float(words / (span_sec / 60.0))


def _segments_to_dicts(raw_segments) -> list[dict]:
    """Normalise segments to plain dicts regardless of library version.

    v0.x returns list[dict] with keys 'text', 'start', 'duration'.
    v1.x returns iterable of objects with .text / .start / .duration attrs.
    """
    out = []
    for s in raw_segments:
        if isinstance(s, dict):
            out.append({"text": s.get("text", ""),
                        "start": s.get("start", 0),
                        "duration": s.get("duration", 0)})
        else:
            out.append({"text": getattr(s, "text", ""),
                        "start": getattr(s, "start", 0),
                        "duration": getattr(s, "duration", 0)})
    return out


def _fetch_raw_v1(video_id: str, languages: tuple[str, ...]) -> Optional[list[dict]]:
    """Fetch using youtube-transcript-api v1.x instance API."""
    api = YouTubeTranscriptApi()
    # list() returns a TranscriptList; find_transcript picks best language
    try:
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript(list(languages))
        fetched = transcript.fetch()
        return _segments_to_dicts(fetched)
    except Exception:
        # Try fetching directly without language preference as fallback
        try:
            fetched = api.fetch(video_id)
            return _segments_to_dicts(fetched)
        except Exception:
            return None


def _fetch_raw_v0(video_id: str, languages: tuple[str, ...]) -> Optional[list[dict]]:
    """Fetch using youtube-transcript-api v0.x class method API."""
    raw = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
    return _segments_to_dicts(raw)


def fetch_one_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> Optional[dict]:
    """Try to fetch captions for a single video.

    Works with youtube-transcript-api v0.x and v1.x automatically.
    """
    if not _HAS_TRANSCRIPT_API:
        return None

    # Serve from on-disk cache when available
    cache = _transcript_path(video_id)
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            text = _flatten_text(data["segments"])
            return {
                "has_captions": True,
                "transcript_source": data.get("source", "api"),
                "transcript_text": text,
                "transcript_word_count": len(text.split()),
                "transcript_lang": data.get("language"),
                "speech_rate_wpm": _compute_speech_rate_wpm(data["segments"], text),
                "transcript_qm_count": text.count("?"),
            }
        except Exception:
            cache.unlink(missing_ok=True)

    # Fetch from YouTube
    try:
        if _YTT_V1:
            segments = _fetch_raw_v1(video_id, languages)
        else:
            segments = _fetch_raw_v0(video_id, languages)
    except Exception as e:
        err = type(e).__name__
        # These exception names appear in both v0 and v1
        if any(x in err for x in ("Disabled", "NotFound", "Unavailable", "NotTranslatable")):
            return {"has_captions": False, "transcript_source": "none"}
        logger.debug("Transcript fetch failed for %s: %s", video_id, err)
        return None

    if not segments:
        return {"has_captions": False, "transcript_source": "none"}

    text = _flatten_text(segments)
    _save_transcript(video_id, source="api", segments=segments, language=languages[0])

    return {
        "has_captions": True,
        "transcript_source": "api",
        "transcript_text": text,
        "transcript_word_count": len(text.split()),
        "transcript_lang": languages[0],
        "speech_rate_wpm": _compute_speech_rate_wpm(segments, text),
        "transcript_qm_count": text.count("?"),
    }


def fetch_transcripts_parallel(records: list[VideoRecord]) -> None:
    """Mutate records in place with transcript features, in parallel."""
    if not records or not _HAS_TRANSCRIPT_API:
        return

    futures = {}
    with ThreadPoolExecutor(max_workers=config.TRANSCRIPT_WORKERS) as pool:
        for rec in records:
            fut = pool.submit(fetch_one_transcript, rec.video_id)
            futures[fut] = rec

        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                logger.warning("Transcript extractor crashed for %s: %s", rec.video_id, e)
                continue

            if result is None:
                continue
            rec.has_captions = result.get("has_captions")
            rec.transcript_source = result.get("transcript_source")
            rec.transcript_text = result.get("transcript_text")
            rec.transcript_word_count = result.get("transcript_word_count")
            rec.transcript_lang = result.get("transcript_lang")
            rec.speech_rate_wpm = result.get("speech_rate_wpm")
            rec.transcript_qm_count = result.get("transcript_qm_count")
