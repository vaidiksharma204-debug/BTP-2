"""
Quota-aware YouTube Data API v3 client.

Design goals:
  - Stay strictly within daily quota (default 10,000 units).
  - Batch requests aggressively (50 IDs per call where possible).
  - Avoid the expensive search.list endpoint (100 units vs 1).
  - Retry transient errors with exponential backoff + jitter.
  - Rotate to fallback API keys when one hits quota.
  - Persist quota usage across runs (so a daily run resumes correctly).

Quota costs (per Google's official table):
    channels.list        →  1 unit per call (≤ 50 channels per call)
    playlistItems.list   →  1 unit per call (≤ 50 items per page)
    videos.list          →  1 unit per call (≤ 50 video IDs per call)
    search.list          →  100 units per call  ← AVOIDED in this codebase

Typical 50,000-video harvest costs ~2,000 quota units total — fits easily
inside a single day's default 10,000 quota.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import date
from typing import Any, Iterator, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from . import config

logger = logging.getLogger(__name__)

_QUOTA_COSTS = {
    "channels.list": 1,
    "playlistItems.list": 1,
    "videos.list": 1,
    "search.list": 100,
}


class QuotaExceeded(Exception):
    """Raised when the YouTube API has signaled (or we've predicted) quota exhaustion."""


class _QuotaTracker:
    """Persists quota usage across runs (one file per date).

    YouTube quota resets at midnight Pacific Time. We key by date so a new
    day starts fresh, while same-day re-runs preserve quota counters.
    """

    def __init__(self, key_id: str):
        self._key_id = key_id
        self._file = config.CHECKPOINTS_DIR / f"quota_{key_id}_{date.today().isoformat()}.json"
        self._used = self._load()

    def _load(self) -> int:
        if self._file.exists():
            try:
                return int(json.loads(self._file.read_text()).get("used", 0))
            except (json.JSONDecodeError, ValueError):
                return 0
        return 0

    def used(self) -> int:
        return self._used

    def remaining(self) -> int:
        return max(0, config.DAILY_QUOTA - self._used)

    def add(self, cost: int) -> None:
        self._used += cost
        self._file.write_text(json.dumps({"used": self._used, "key_id": self._key_id}))


class YouTubeClient:
    """Quota-aware wrapper around the YouTube Data API v3."""

    def __init__(self, api_keys: Optional[list[str]] = None):
        keys = api_keys or [config.YOUTUBE_API_KEY] + config.YOUTUBE_API_KEY_FALLBACKS
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError(
                "No YouTube API key configured. Set YOUTUBE_API_KEY in your .env file."
            )
        self._keys = keys
        self._key_idx = 0
        self._tracker = _QuotaTracker(self._key_id())
        self._client = self._make_client()
        logger.info(
            "YouTubeClient ready · key index %d · %d/%d quota used today",
            self._key_idx, self._tracker.used(), config.DAILY_QUOTA
        )

    def _key_id(self) -> str:
        """Short, non-sensitive identifier for the active key."""
        k = self._keys[self._key_idx]
        return f"{k[:6]}_{k[-4:]}" if len(k) >= 10 else f"key{self._key_idx}"

    def _make_client(self):
        return build(
            "youtube", "v3",
            developerKey=self._keys[self._key_idx],
            cache_discovery=False,
        )

    def _rotate_key(self) -> bool:
        """Switch to next fallback key. Returns True on success."""
        if self._key_idx + 1 >= len(self._keys):
            return False
        self._key_idx += 1
        self._tracker = _QuotaTracker(self._key_id())
        self._client = self._make_client()
        logger.warning("Rotated to fallback API key (index %d)", self._key_idx)
        return True

    def _check_quota(self, op: str) -> int:
        cost = _QUOTA_COSTS.get(op, 1)
        if self._tracker.used() + cost > config.DAILY_QUOTA - config.QUOTA_SAFETY_MARGIN:
            if not self._rotate_key():
                raise QuotaExceeded(
                    f"Quota nearly exhausted: {self._tracker.used()}/{config.DAILY_QUOTA} used. "
                    f"Add a fallback key or wait for daily reset (00:00 PT)."
                )
        return cost

    def _execute(self, request, op_name: str):
        """Execute a built request with retry, backoff, and quota tracking."""
        cost = self._check_quota(op_name)

        for attempt in range(config.MAX_RETRIES):
            try:
                resp = request.execute()
                self._tracker.add(cost)
                return resp

            except HttpError as e:
                status = getattr(e.resp, "status", 0)
                err_text = str(e)

                if status == 403 and ("quotaExceeded" in err_text or "rateLimitExceeded" in err_text):
                    logger.warning("Quota / rate limit hit on %s: %s", op_name, err_text[:200])
                    if self._rotate_key():
                        raise QuotaExceeded("Rotated to fallback key — caller should retry") from e
                    raise QuotaExceeded("All API keys exhausted") from e

                if status in (429,) or 500 <= status < 600:
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.info(
                        "Transient %s on %s (attempt %d/%d) — sleeping %.1fs",
                        status, op_name, attempt + 1, config.MAX_RETRIES, delay
                    )
                    time.sleep(delay)
                    continue

                logger.error("Non-retryable error on %s: status=%s body=%s",
                             op_name, status, err_text[:200])
                raise

            except Exception as e:
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("Network error on %s (attempt %d) — sleeping %.1fs: %s",
                               op_name, attempt + 1, delay, type(e).__name__)
                time.sleep(delay)

        raise RuntimeError(f"{op_name} failed after {config.MAX_RETRIES} retries")

    # ─── public API ─────────────────────────────────────────────────────────

    def quota_remaining(self) -> int:
        return self._tracker.remaining()

    def get_channels(self, channel_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full channel records (50 per call). Missing IDs silently dropped."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(channel_ids), config.CHANNEL_BATCH_SIZE):
            chunk = channel_ids[i:i + config.CHANNEL_BATCH_SIZE]
            req = self._client.channels().list(
                part="snippet,statistics,contentDetails,topicDetails,brandingSettings",
                id=",".join(chunk),
                maxResults=config.CHANNEL_BATCH_SIZE,
            )
            try:
                resp = self._execute(req, "channels.list")
                results.extend(resp.get("items", []))
            except QuotaExceeded:
                req = self._client.channels().list(
                    part="snippet,statistics,contentDetails,topicDetails,brandingSettings",
                    id=",".join(chunk),
                    maxResults=config.CHANNEL_BATCH_SIZE,
                )
                resp = self._execute(req, "channels.list")
                results.extend(resp.get("items", []))
        return results

    def iter_uploads(self, uploads_playlist_id: str, max_videos: int = 100) -> Iterator[str]:
        """Yield video IDs from uploads playlist, newest first."""
        page_token: Optional[str] = None
        yielded = 0

        while yielded < max_videos:
            page_size = min(50, max_videos - yielded)
            req = self._client.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=page_size,
                pageToken=page_token,
            )
            try:
                resp = self._execute(req, "playlistItems.list")
            except QuotaExceeded:
                req = self._client.playlistItems().list(
                    part="contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=page_size,
                    pageToken=page_token,
                )
                resp = self._execute(req, "playlistItems.list")

            for item in resp.get("items", []):
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    yield vid
                    yielded += 1
                    if yielded >= max_videos:
                        return

            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    def get_videos(self, video_ids: list[str]) -> list[dict[str, Any]]:
        """Batch-fetch full video metadata (50 per call)."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(video_ids), config.VIDEO_BATCH_SIZE):
            chunk = video_ids[i:i + config.VIDEO_BATCH_SIZE]
            req = self._client.videos().list(
                part="snippet,statistics,contentDetails,status,topicDetails",
                id=",".join(chunk),
                maxResults=config.VIDEO_BATCH_SIZE,
            )
            try:
                resp = self._execute(req, "videos.list")
                results.extend(resp.get("items", []))
            except QuotaExceeded:
                req = self._client.videos().list(
                    part="snippet,statistics,contentDetails,status,topicDetails",
                    id=",".join(chunk),
                    maxResults=config.VIDEO_BATCH_SIZE,
                )
                resp = self._execute(req, "videos.list")
                results.extend(resp.get("items", []))
        return results
