"""
End-to-end harvest orchestrator.

Pipeline (per channel):
    1. Resolve channel record (uploads playlist ID + Group B fields)
    2. Page through uploads playlist, collecting video IDs
    3. Filter to IDs not already harvested (checkpoint)
    4. Batch-fetch full video metadata (50 IDs per API call)
    5. Parse API responses into VideoRecord objects
    6. Apply duration filter
    7. Download thumbnails + extract basic visual features (parallel)
    8. Fetch transcripts via youtube-transcript-api (parallel)
    9. Compute Group C engineered text features (parallel)
   10. Write a parquet shard for the channel
   11. Update checkpoint
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from . import config
from .feature_engineer import engineer_batch
from .parser import parse_video
from .schema import VideoRecord
from .storage import Checkpoint, write_shard, consolidate_shards
from .thumbnail_extractor import fetch_thumbnails_parallel
from .transcript_extractor import fetch_transcripts_parallel
from .youtube_client import QuotaExceeded, YouTubeClient

logger = logging.getLogger(__name__)


def _filter_by_duration(records: list[VideoRecord]) -> list[VideoRecord]:
    kept = []
    for r in records:
        d = r.duration_seconds
        if d is None:
            kept.append(r)
            continue
        if config.MIN_VIDEO_DURATION_SEC <= d <= config.MAX_VIDEO_DURATION_SEC:
            kept.append(r)
    dropped = len(records) - len(kept)
    if dropped:
        logger.debug("Dropped %d records by duration filter", dropped)
    return kept


def harvest_channel(
    yt: YouTubeClient,
    channel_id: str,
    max_videos: int,
    checkpoint: Checkpoint,
    sector: str | None = None,
) -> int:
    """Harvest one channel end-to-end. Returns new records persisted."""
    logger.info("───── Channel: %s (sector=%s) ─────", channel_id, sector)

    channels = yt.get_channels([channel_id])
    if not channels:
        logger.warning("Channel %s not found / inaccessible", channel_id)
        return 0
    channel = channels[0]

    uploads_pid = (
        channel.get("contentDetails", {})
               .get("relatedPlaylists", {})
               .get("uploads")
    )
    if not uploads_pid:
        logger.warning("Channel %s has no uploads playlist", channel_id)
        return 0

    all_ids: list[str] = list(yt.iter_uploads(uploads_pid, max_videos=max_videos))
    logger.info("  fetched %d video IDs from uploads", len(all_ids))
    if not all_ids:
        return 0

    new_ids = checkpoint.filter_new(all_ids)
    skipped = len(all_ids) - len(new_ids)
    if skipped:
        logger.info("  skipping %d already-harvested IDs", skipped)
    if not new_ids:
        return 0

    api_items = yt.get_videos(new_ids)
    logger.info("  fetched metadata for %d/%d IDs (others may be deleted/private)",
                len(api_items), len(new_ids))

    records: list[VideoRecord] = []
    for item in api_items:
        rec = parse_video(item, channel_record=channel)
        if rec is None:
            continue
        if sector:
            rec.sector = sector
        records.append(rec)
    logger.info("  parsed %d records", len(records))

    records = _filter_by_duration(records)
    if not records:
        return 0

    t0 = time.time()
    fetch_thumbnails_parallel(records)
    logger.info("  thumbnails done in %.1fs", time.time() - t0)

    t0 = time.time()
    fetch_transcripts_parallel(records)
    logger.info("  transcripts done in %.1fs", time.time() - t0)

    t0 = time.time()
    engineer_batch(records)
    logger.info("  feature engineering done in %.1fs", time.time() - t0)

    write_shard(records, channel_id)

    checkpoint.add_many(r.video_id for r in records)
    logger.info("  ✓ persisted %d records (running total: %d)",
                len(records), checkpoint.count())
    return len(records)


def harvest_from_seed(
    seed_csv_path: str,
    max_videos_per_channel: int = 100,
    consolidate: bool = True,
) -> int:
    """Run a full harvest pass from a seed-channel CSV."""
    df = pd.read_csv(seed_csv_path)
    if "channel_id" not in df.columns:
        raise ValueError("Seed CSV must have a 'channel_id' column")

    yt = YouTubeClient()
    checkpoint = Checkpoint()

    total_new = 0
    for i, row in df.iterrows():
        channel_id = str(row["channel_id"]).strip()
        if not channel_id:
            continue
        sector = str(row.get("sector", "")).strip() or None

        if yt.quota_remaining() < config.QUOTA_SAFETY_MARGIN:
            logger.warning(
                "Quota nearly exhausted (%d remaining). Stopping at row %d/%d.",
                yt.quota_remaining(), i, len(df),
            )
            break

        try:
            n = harvest_channel(
                yt, channel_id,
                max_videos=max_videos_per_channel,
                checkpoint=checkpoint,
                sector=sector,
            )
            total_new += n
        except QuotaExceeded as e:
            logger.warning("Quota exceeded mid-run: %s", e)
            break
        except Exception as e:
            logger.exception("Channel %s failed: %s", channel_id, e)
            continue

    logger.info("═════ Run complete ═════")
    logger.info("New records this run: %d", total_new)
    logger.info("Total in checkpoint: %d", checkpoint.count())
    logger.info("Quota used: %d / %d",
                config.DAILY_QUOTA - yt.quota_remaining(), config.DAILY_QUOTA)

    if consolidate and total_new > 0:
        try:
            out = consolidate_shards()
            logger.info("Consolidated dataset → %s", out)
        except Exception as e:
            logger.warning("Consolidation failed: %s", e)

    return total_new
