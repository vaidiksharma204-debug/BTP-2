"""
Central configuration for the BTP-2 YouTube scraper.

Loads sensitive values from environment variables (.env file).
All paths and tunables live here so the rest of the code stays clean.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ─── API keys ───────────────────────────────────────────────────────────────
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

_FALLBACKS_RAW = os.getenv("YOUTUBE_API_KEY_FALLBACKS", "")
YOUTUBE_API_KEY_FALLBACKS: list[str] = [k.strip() for k in _FALLBACKS_RAW.split(",") if k.strip()]

# ─── Quota management ───────────────────────────────────────────────────────
DAILY_QUOTA: int = int(os.getenv("YOUTUBE_DAILY_QUOTA", "10000"))
QUOTA_SAFETY_MARGIN: int = 200

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LOGS_DIR = DATA_DIR / "logs"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, THUMBNAILS_DIR, TRANSCRIPTS_DIR, CHECKPOINTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Concurrency tuning ─────────────────────────────────────────────────────
THUMBNAIL_WORKERS: int = 24
TRANSCRIPT_WORKERS: int = 12
FEATURE_WORKERS: int = 8

# ─── Batch sizes ────────────────────────────────────────────────────────────
VIDEO_BATCH_SIZE: int = 50
CHANNEL_BATCH_SIZE: int = 50
PERSIST_BATCH_SIZE: int = 100

# ─── Whisper (optional) ─────────────────────────────────────────────────────
WHISPER_MODEL: str = "base"
WHISPER_MAX_SECONDS: int = 120

# ─── Filters ────────────────────────────────────────────────────────────────
MIN_VIDEO_DURATION_SEC: int = 10
MAX_VIDEO_DURATION_SEC: int = 7200
TARGET_LANGUAGES: list[str] = ["en"]

# ─── Network tuning ─────────────────────────────────────────────────────────
HTTP_TIMEOUT_SEC: int = 20
MAX_RETRIES: int = 5
USER_AGENT: str = "BTP2-Scraper/1.0 (academic research; contact: btp2@iitkgp.ac.in)"
