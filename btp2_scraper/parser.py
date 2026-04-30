"""
Parser: converts raw YouTube API responses into VideoRecord objects.

Responsibilities:
  - Pull every available field from the API response into the schema.
  - Parse ISO 8601 durations into seconds.
  - Compute basic derived metrics (engagement rate, views/sub, etc.)
  - Extract hashtags from title + description.
  - NEVER raise on missing fields — leave them as None and let downstream
    feature engineer fill in what it can.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .schema import VideoRecord, now_iso

logger = logging.getLogger(__name__)

_DURATION_RE = re.compile(
    r"^PT"
    r"(?:(?P<hours>\d+)H)?"
    r"(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+)S)?$"
)

_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_\u00C0-\u024F\u0900-\u097F]+)")


def parse_iso_duration(iso: str) -> Optional[int]:
    """Convert 'PT1H23M45S' → 5025 (seconds). Returns None on failure."""
    if not iso:
        return None
    m = _DURATION_RE.match(iso)
    if not m:
        return None
    h = int(m.group("hours") or 0)
    mi = int(m.group("minutes") or 0)
    s = int(m.group("seconds") or 0)
    return h * 3600 + mi * 60 + s


def extract_hashtags(text: str) -> list[str]:
    """Pull hashtags from text. Lowercased, deduped, original order."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _HASHTAG_RE.finditer(text):
        tag = m.group(1).lower()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _safe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_video(api_item: dict, channel_record: Optional[dict] = None) -> Optional[VideoRecord]:
    """Convert one videos.list item into a VideoRecord.

    Returns None if the item is too malformed to be useful.
    """
    if not api_item or "id" not in api_item:
        return None

    snip = api_item.get("snippet", {}) or {}
    stats = api_item.get("statistics", {}) or {}
    content = api_item.get("contentDetails", {}) or {}

    video_id = api_item["id"]
    channel_id = snip.get("channelId", "")
    if not channel_id:
        logger.debug("Skipping %s: missing channel_id", video_id)
        return None

    # Temporal
    published_at = snip.get("publishedAt")
    upload_dt: Optional[datetime] = None
    if published_at:
        try:
            upload_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            upload_dt = None

    rec = VideoRecord(
        video_id=video_id,
        channel_id=channel_id,
        title=snip.get("title", "") or "",
        description=snip.get("description", "") or "",
        upload_date=published_at,
        scraped_date=now_iso(),
        language=(snip.get("defaultAudioLanguage") or snip.get("defaultLanguage") or "")[:2] or None,
        youtube_category_id=_safe_int(snip.get("categoryId")),
    )

    # Hashtags
    raw_text = f"{rec.title} {rec.description}"
    rec.hashtags_normalized = extract_hashtags(raw_text)
    rec.existing_hashtags = list(rec.hashtags_normalized)
    rec.hashtag_count = len(rec.hashtags_normalized)

    # Duration
    rec.duration_seconds = parse_iso_duration(content.get("duration", ""))

    # Performance
    views = _safe_int(stats.get("viewCount"))
    likes = _safe_int(stats.get("likeCount"))
    comments = _safe_int(stats.get("commentCount"))
    favs = _safe_int(stats.get("favoriteCount"))
    rec.views_lifetime = views
    rec.likes_count = likes
    rec.comments_count = comments
    rec.favorite_count = favs

    if views and views > 0:
        rec.likes_per_view = (likes or 0) / views
        rec.comments_per_view = (comments or 0) / views
        rec.engagement_rate = ((likes or 0) + (comments or 0)) / views * 100.0

    # Temporal features
    if upload_dt is not None:
        if upload_dt.tzinfo is None:
            upload_dt = upload_dt.replace(tzinfo=timezone.utc)
        else:
            upload_dt = upload_dt.astimezone(timezone.utc)
        rec.upload_hour_utc = upload_dt.hour
        rec.upload_dow = upload_dt.weekday()
        rec.upload_month = upload_dt.month
        rec.upload_week_of_year = int(upload_dt.strftime("%V"))
        rec.season = f"Q{(upload_dt.month - 1) // 3 + 1}"

    # Thumbnail (highest available)
    thumbs = (snip.get("thumbnails") or {})
    for key in ("maxres", "standard", "high", "medium", "default"):
        if key in thumbs and thumbs[key].get("url"):
            rec.thumbnail_url = thumbs[key]["url"]
            rec.thumb_width = _safe_int(thumbs[key].get("width"))
            rec.thumb_height = _safe_int(thumbs[key].get("height"))
            break

    # Channel context (Group B)
    if channel_record is not None:
        _enrich_with_channel(rec, channel_record)

    return rec


def _enrich_with_channel(rec: VideoRecord, channel: dict) -> None:
    """Fill Group B (channel context) fields from a channel record."""
    snip = channel.get("snippet", {}) or {}
    stats = channel.get("statistics", {}) or {}
    topics = channel.get("topicDetails", {}) or {}

    rec.channel_subscribers = _safe_int(stats.get("subscriberCount"))
    rec.channel_total_views = _safe_int(stats.get("viewCount"))
    rec.channel_total_videos = _safe_int(stats.get("videoCount"))
    rec.channel_country = snip.get("country")
    rec.channel_topic_categories = list(topics.get("topicCategories") or [])

    created = snip.get("publishedAt")
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            rec.channel_age_days = max(0, (now_dt - created_dt).days)
        except ValueError:
            pass

    if rec.channel_subscribers and rec.channel_subscribers > 0 and rec.views_lifetime:
        rec.views_per_subscriber = rec.views_lifetime / rec.channel_subscribers
