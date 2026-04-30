#!/usr/bin/env python3
"""
Command-line entry point for the BTP-2 scraper.

Examples:

  # Run a full harvest from a seed CSV (most common use):
  python run.py --seed seed_channels.csv --max-per-channel 100

  # Just harvest one channel by ID:
  python run.py --channel UCXuqSBlHAE6Xw-yeJA0Tunw --max-per-channel 50

  # Consolidate already-written shards into the final parquet:
  python run.py --consolidate-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from btp2_scraper import config
from btp2_scraper.pipeline import harvest_from_seed, harvest_channel
from btp2_scraper.storage import Checkpoint, consolidate_shards
from btp2_scraper.youtube_client import YouTubeClient


def setup_logging(verbose: bool) -> None:
    """Send INFO+ logs to console and DEBUG+ logs to a dated file."""
    level = logging.DEBUG if verbose else logging.INFO
    log_file = config.LOGS_DIR / f"harvest_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Wipe any prior handlers (re-runs in the same process)
    root.handlers = []

    # Console handler — terse format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt=datefmt))
    root.addHandler(ch)

    # File handler — verbose, persistent
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    # Quieten googleapiclient (very chatty at DEBUG)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> int:
    p = argparse.ArgumentParser(description="BTP-2 YouTube data harvester")
    p.add_argument("--seed", type=Path, help="Path to seed CSV (channel_id, sector, ...)")
    p.add_argument("--channel", type=str, help="Harvest a single channel by ID")
    p.add_argument("--max-per-channel", type=int, default=100,
                   help="Max videos to fetch per channel (default: 100)")
    p.add_argument("--no-consolidate", action="store_true",
                   help="Skip the final shard consolidation step")
    p.add_argument("--consolidate-only", action="store_true",
                   help="Only consolidate existing shards; don't fetch anything new")
    p.add_argument("-v", "--verbose", action="store_true", help="Show DEBUG-level logs on console")
    args = p.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("run")

    if args.consolidate_only:
        path = consolidate_shards()
        log.info("Consolidated → %s", path)
        return 0

    if args.channel:
        yt = YouTubeClient()
        cp = Checkpoint()
        n = harvest_channel(yt, args.channel,
                            max_videos=args.max_per_channel,
                            checkpoint=cp)
        log.info("Single-channel harvest done — %d new records", n)
        return 0

    if args.seed:
        if not args.seed.exists():
            log.error("Seed file not found: %s", args.seed)
            return 1
        n = harvest_from_seed(
            str(args.seed),
            max_videos_per_channel=args.max_per_channel,
            consolidate=not args.no_consolidate,
        )
        log.info("Seed-driven harvest done — %d new records", n)
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
