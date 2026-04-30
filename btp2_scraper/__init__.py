"""BTP-2 YouTube data scraper."""

__version__ = "1.0.0"

from .pipeline import harvest_channel, harvest_from_seed
from .schema import VideoRecord
from .youtube_client import YouTubeClient, QuotaExceeded

__all__ = [
    "harvest_channel",
    "harvest_from_seed",
    "VideoRecord",
    "YouTubeClient",
    "QuotaExceeded",
]
