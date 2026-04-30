"""
Storage: persist VideoRecords to parquet shards and track resume state.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config
from .schema import VideoRecord

logger = logging.getLogger(__name__)


_CHECKPOINT_FILE = config.CHECKPOINTS_DIR / "harvested.json"


class Checkpoint:
    """Tracks which video IDs have already been fully processed and persisted."""

    def __init__(self):
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if _CHECKPOINT_FILE.exists():
            try:
                data = json.loads(_CHECKPOINT_FILE.read_text())
                self._ids = set(data.get("video_ids", []))
                logger.info("Loaded checkpoint with %d already-harvested videos", len(self._ids))
            except json.JSONDecodeError:
                logger.warning("Checkpoint file corrupt; starting fresh")

    def has(self, video_id: str) -> bool:
        return video_id in self._ids

    def filter_new(self, ids: Iterable[str]) -> list[str]:
        return [i for i in ids if i not in self._ids]

    def add_many(self, ids: Iterable[str]) -> None:
        self._ids.update(ids)
        self._save()

    def _save(self) -> None:
        _CHECKPOINT_FILE.write_text(json.dumps({
            "video_ids": sorted(self._ids),
            "count": len(self._ids),
        }))

    def count(self) -> int:
        return len(self._ids)


def _records_to_dataframe(records: list[VideoRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [r.to_dict() for r in records]
    return pd.DataFrame(rows)


def write_shard(records: list[VideoRecord], channel_id: str) -> Path | None:
    """Write a batch of records to a single parquet shard."""
    if not records:
        return None

    df = _records_to_dataframe(records)
    today = date.today().isoformat()
    fname = f"{channel_id}__{today}__{len(records)}.parquet"
    path = config.PROCESSED_DIR / fname

    counter = 1
    while path.exists():
        fname = f"{channel_id}__{today}__{len(records)}__{counter}.parquet"
        path = config.PROCESSED_DIR / fname
        counter += 1

    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    logger.info("Wrote %d records → %s", len(records), path.name)
    return path


def consolidate_shards(output_path: Path | None = None) -> Path:
    """Concatenate all per-channel shards into a single parquet file."""
    if output_path is None:
        output_path = config.DATA_DIR / "btp2_v1.parquet"

    shards = sorted(config.PROCESSED_DIR.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No shards found in {config.PROCESSED_DIR}")

    logger.info("Consolidating %d shards → %s", len(shards), output_path.name)
    dfs = [pd.read_parquet(s, engine="pyarrow") for s in shards]
    full = pd.concat(dfs, ignore_index=True)

    if "scraped_date" in full.columns:
        full = full.sort_values("scraped_date").drop_duplicates(
            subset=["video_id"], keep="last"
        )
    else:
        full = full.drop_duplicates(subset=["video_id"], keep="last")

    full.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")
    logger.info("Wrote %d unique videos → %s", len(full), output_path)
    return output_path
