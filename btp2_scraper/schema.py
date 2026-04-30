"""
Dataset schema: a single row in the BTP-2 dataset.

Fields are organized in 8 groups matching the BTP-2 dataset specification:
    A — Identifiers & source metadata
    B — Channel context
    C — Pre-publication text features
    D — Thumbnail visual features (basic; embeddings done in a later pipeline)
    E — Audio & transcript features
    F — Temporal & trend features
    G — Performance KPIs (training targets)
    H — Hashtag targets
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class VideoRecord:
    # ── Group A: Identifiers ────────────────────────────────────────────────
    video_id: str
    channel_id: str
    title: str
    description: str
    upload_date: Optional[str] = None
    scraped_date: Optional[str] = None
    region_code: Optional[str] = None
    language: Optional[str] = None
    youtube_category_id: Optional[int] = None
    fine_category: Optional[str] = None
    sector: Optional[str] = None

    # ── Group B: Channel context ────────────────────────────────────────────
    channel_subscribers: Optional[int] = None
    channel_age_days: Optional[int] = None
    channel_total_views: Optional[int] = None
    channel_total_videos: Optional[int] = None
    channel_country: Optional[str] = None
    channel_topic_categories: list[str] = field(default_factory=list)
    channel_avg_views_30d: Optional[float] = None
    channel_avg_engagement_30d: Optional[float] = None
    channel_upload_frequency: Optional[float] = None

    # ── Group C: Pre-publication text features ──────────────────────────────
    title_length_chars: Optional[int] = None
    title_word_count: Optional[int] = None
    title_sentiment_polarity: Optional[float] = None
    title_subjectivity: Optional[float] = None
    title_has_number: Optional[bool] = None
    title_has_question: Optional[bool] = None
    title_caps_ratio: Optional[float] = None
    title_emoji_count: Optional[int] = None
    description_length_chars: Optional[int] = None
    description_word_count: Optional[int] = None
    description_sentiment_polarity: Optional[float] = None
    description_subjectivity: Optional[float] = None
    description_url_count: Optional[int] = None
    combined_sentiment: Optional[float] = None
    flesch_readability: Optional[float] = None
    cta_word_count: Optional[int] = None
    cta_presence: Optional[bool] = None
    existing_hashtags: list[str] = field(default_factory=list)
    hashtag_count: Optional[int] = None

    # ── Group D: Thumbnail features (basic) ─────────────────────────────────
    thumbnail_url: Optional[str] = None
    thumbnail_local_path: Optional[str] = None
    thumb_width: Optional[int] = None
    thumb_height: Optional[int] = None
    thumb_brightness: Optional[float] = None
    thumb_contrast: Optional[float] = None
    thumb_saturation: Optional[float] = None
    thumb_dominant_colors: list[str] = field(default_factory=list)
    thumb_face_count: Optional[int] = None
    thumb_has_text: Optional[bool] = None
    thumb_text_word_count: Optional[int] = None

    # ── Group E: Audio / transcript ─────────────────────────────────────────
    has_captions: Optional[bool] = None
    transcript_source: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_word_count: Optional[int] = None
    speech_rate_wpm: Optional[float] = None
    transcript_lang: Optional[str] = None
    transcript_qm_count: Optional[int] = None

    # ── Group F: Temporal / trend ───────────────────────────────────────────
    upload_hour_utc: Optional[int] = None
    upload_dow: Optional[int] = None
    upload_month: Optional[int] = None
    upload_week_of_year: Optional[int] = None
    is_holiday: Optional[bool] = None
    season: Optional[str] = None

    # ── Group G: Performance KPIs (training targets) ────────────────────────
    duration_seconds: Optional[int] = None
    views_lifetime: Optional[int] = None
    likes_count: Optional[int] = None
    comments_count: Optional[int] = None
    favorite_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    views_per_subscriber: Optional[float] = None
    likes_per_view: Optional[float] = None
    comments_per_view: Optional[float] = None
    engagement_tier: Optional[str] = None
    views_tier: Optional[str] = None

    # ── Group H: Hashtag targets ────────────────────────────────────────────
    hashtags_normalized: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso() -> str:
    """UTC timestamp in ISO-8601, suitable for the scraped_date field."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
