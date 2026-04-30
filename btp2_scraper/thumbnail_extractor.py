"""
Thumbnail downloader and basic visual feature extractor.

What this module does:
  - Download the highest-quality thumbnail per video (parallel, rate-limited).
  - Compute lightweight visual features inline:
      * brightness, contrast, saturation
      * 5 dominant colors (k-means on downsampled image)
  - Save the JPEG to disk for later (heavier) processing
    (face detection, OCR, EfficientNet embeddings) in a separate pipeline.

Why split heavy features out:
  Face detection / OCR / CNN embeddings are GPU-bound and slow. Keeping
  the harvest pipeline lean means we can ingest 50k videos in hours, not
  days. Heavier features are added in a post-processing pass that operates
  on already-downloaded thumbnails.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .schema import VideoRecord

logger = logging.getLogger(__name__)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        pool_connections=config.THUMBNAIL_WORKERS,
        pool_maxsize=config.THUMBNAIL_WORKERS,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_FALLBACK_KEYS = ("maxresdefault", "sddefault", "hqdefault", "mqdefault", "default")


def _build_fallback_urls(video_id: str) -> list[str]:
    return [f"https://i.ytimg.com/vi/{video_id}/{key}.jpg" for key in _FALLBACK_KEYS]


def _download_one(session: requests.Session, url: str) -> Optional[bytes]:
    try:
        r = session.get(url, timeout=config.HTTP_TIMEOUT_SEC)
        if r.status_code == 200 and len(r.content) > 1024:
            return r.content
    except requests.RequestException as e:
        logger.debug("Thumbnail %s failed: %s", url, type(e).__name__)
    return None


def _save_thumbnail(image_bytes: bytes, path: Path) -> bool:
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        img.save(path, format="JPEG", quality=88, optimize=True)
        return True
    except Exception as e:
        logger.debug("Failed to save thumbnail %s: %s", path.name, e)
        return False


def _compute_visual_features(image_bytes: bytes) -> dict:
    """Brightness, contrast, saturation, and 5 dominant colors."""
    out = {
        "brightness": None,
        "contrast": None,
        "saturation": None,
        "dominant_colors": [],
    }
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        small = img.resize((160, 90), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)

        # Brightness via Rec. 709 luminance, normalized to 0..1
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        out["brightness"] = float(lum.mean() / 255.0)
        out["contrast"] = float(lum.std() / 255.0)

        # Saturation: mean S in HSV
        rgb_norm = arr / 255.0
        cmax = rgb_norm.max(axis=-1)
        cmin = rgb_norm.min(axis=-1)
        delta = cmax - cmin
        sat = np.where(cmax > 0, delta / np.maximum(cmax, 1e-6), 0.0)
        out["saturation"] = float(sat.mean())

        # Dominant colors via mini k-means
        pixels = arr.reshape(-1, 3)
        rng = np.random.default_rng(42)
        if pixels.shape[0] > 2000:
            idx = rng.choice(pixels.shape[0], size=2000, replace=False)
            pixels = pixels[idx]

        centers = _mini_kmeans(pixels, k=5, iters=4, rng=rng)
        labels = _assign_labels(pixels, centers)
        counts = np.bincount(labels, minlength=5)
        order = np.argsort(-counts)
        hex_colors = []
        for i in order:
            r, g, b = centers[i].clip(0, 255).astype(int)
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        out["dominant_colors"] = hex_colors

    except Exception as e:
        logger.debug("Visual features failed: %s", e)

    return out


def _mini_kmeans(pixels: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    n = pixels.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centers = pixels[init_idx].astype(np.float32)
    for _ in range(iters):
        labels = _assign_labels(pixels, centers)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = pixels[mask].mean(axis=0)
    return centers


def _assign_labels(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return d.argmin(axis=-1)


def fetch_one_thumbnail(video_id: str, session: requests.Session) -> Optional[dict]:
    """Download + analyze the best thumbnail for one video."""
    out_path = config.THUMBNAILS_DIR / f"{video_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 1024:
        try:
            return {
                "thumbnail_local_path": str(out_path),
                **_compute_visual_features(out_path.read_bytes()),
            }
        except Exception:
            out_path.unlink(missing_ok=True)

    for url in _build_fallback_urls(video_id):
        data = _download_one(session, url)
        if data and _save_thumbnail(data, out_path):
            return {
                "thumbnail_local_path": str(out_path),
                **_compute_visual_features(data),
            }
    return None


def fetch_thumbnails_parallel(records: list[VideoRecord]) -> None:
    """Mutate records in place with thumbnail features, in parallel."""
    if not records:
        return

    session = _make_session()
    futures = {}

    with ThreadPoolExecutor(max_workers=config.THUMBNAIL_WORKERS) as pool:
        for rec in records:
            fut = pool.submit(fetch_one_thumbnail, rec.video_id, session)
            futures[fut] = rec

        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                logger.warning("Thumbnail extractor crashed for %s: %s", rec.video_id, e)
                result = None

            if result:
                rec.thumbnail_local_path = result.get("thumbnail_local_path")
                rec.thumb_brightness = result.get("brightness")
                rec.thumb_contrast = result.get("contrast")
                rec.thumb_saturation = result.get("saturation")
                rec.thumb_dominant_colors = result.get("dominant_colors") or []

    session.close()
