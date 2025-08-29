"""Enhanced caching utilities with async support."""

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, TypeVar

import aiofiles
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class AsyncFileCache:
    """Async file-based cache for DataFrame and other serializable data."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or settings.cache_directory
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a stable string representation
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, cache_key: str, suffix: str = ".pkl") -> str:
        """Get full cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}{suffix}")

    async def get(self, cache_key: str, max_age_seconds: int = 3600) -> Any:
        """Get cached value if it exists and is fresh."""
        cache_path = self._cache_path(cache_key)

        try:
            if not os.path.exists(cache_path):
                return None

            stat = os.stat(cache_path)
            age = time.time() - stat.st_mtime

            if age > max_age_seconds:
                logger.debug("Cache expired for %s (age: %.1fs)", cache_key, age)
                return None

            # Use pandas for DataFrame objects, pickle for others
            if cache_path.endswith('.parquet'):
                return pd.read_parquet(cache_path)
            else:
                return pd.read_pickle(cache_path)

        except Exception as e:
            logger.warning("Failed to load from cache %s: %s", cache_key, e)
            return None

    async def set(self, cache_key: str, value: Any) -> None:
        """Store value in cache."""
        try:
            if isinstance(value, pd.DataFrame):
                # Use parquet for DataFrames - better performance and compression
                cache_path = self._cache_path(cache_key, ".parquet")
                value.to_parquet(cache_path)
            else:
                # Use pickle for other objects
                cache_path = self._cache_path(cache_key, ".pkl")
                value.to_pickle(cache_path)

            logger.debug("Cached data to %s", cache_key)
        except Exception as e:
            logger.warning("Failed to cache %s: %s", cache_key, e)

    def cache_async(
        self,
        max_age_seconds: int = 3600,
        key_func: Callable[..., str] | None = None
    ) -> Callable[[F], F]:
        """Decorator for async function caching."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._cache_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = await self.get(cache_key, max_age_seconds)
                if cached_result is not None:
                    logger.debug("Cache hit for %s", func.__name__)
                    return cached_result

                # Execute function
                logger.debug("Cache miss for %s, executing function", func.__name__)
                result = await func(*args, **kwargs)

                # Cache result
                await self.set(cache_key, result)

                return result
            return wrapper
        return decorator


# Global cache instance
_cache: AsyncFileCache | None = None


def get_cache() -> AsyncFileCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = AsyncFileCache()
    return _cache


def cached_async(
    max_age_seconds: int = 3600,
    key_func: Callable[..., str] | None = None
):
    """Convenience decorator for async caching."""
    return get_cache().cache_async(max_age_seconds, key_func)
