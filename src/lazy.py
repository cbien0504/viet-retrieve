"""Nạp model lười (lazy) và giải phóng bộ nhớ sau khi dùng."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def current_rss_mb() -> float:
    """Bộ nhớ thường trú (RSS) hiện tại của tiến trình, đơn vị MiB. Chỉ hỗ trợ Linux (nơi khác trả về nan)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return float("nan")


def _empty_accelerator_cache() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _return_freed_memory_to_os() -> None:
    """glibc giữ lại vùng nhớ đã free trong arena; ``malloc_trim(0)`` trả phần trống về hệ điều hành.

    Chỉ có tác dụng trên Linux/glibc; ở nơi khác bỏ qua.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # pragma: no cover - không phải glibc
        pass


def quiet_model_loading() -> None:
    """Tắt thanh tiến trình và cảnh báo thông tin của transformers / huggingface_hub khi nạp model.

    Lỗi thật (không tải được model...) vẫn được ném ra dưới dạng exception.
    """
    try:
        from huggingface_hub.utils import disable_progress_bars
        from transformers.utils import logging as hf_logging

        disable_progress_bars()
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    except ImportError:  # pragma: no cover
        pass


class LazyModel(Generic[T]):
    """Giữ một model nặng: chỉ nạp ở lần dùng đầu tiên, có thể giải phóng khi không cần.

    - ``run(fn)`` nạp model nếu chưa có, gọi ``fn(model)``; nếu ``release_after_use`` thì giải phóng ngay sau đó.
    - ``hold()`` giữ model trong suốt khối ``with`` (ví dụ một batch truy vấn khi đánh giá) để không
      phải nạp lại sau mỗi truy vấn; model được giải phóng khi rời khối nếu ``release_after_use``.
    """

    def __init__(self, name: str, loader: Callable[[], T], release_after_use: bool = False) -> None:
        self.name = name
        self.release_after_use = release_after_use
        self._loader = loader
        self._model: T | None = None
        self._holds = 0
        self._lock = threading.RLock()
        self.load_count = 0
        self.last_load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get(self) -> T:
        """Trả về model, nạp nếu cần."""
        with self._lock:
            if self._model is None:
                started = time.perf_counter()
                self._model = self._loader()
                self.last_load_seconds = time.perf_counter() - started
                self.load_count += 1
                logger.info("Đã nạp %s trong %.2fs", self.name, self.last_load_seconds)
            return self._model

    def release(self) -> None:
        """Bỏ tham chiếu tới model và thu gom bộ nhớ."""
        with self._lock:
            if self._model is None:
                return
            self._model = None
        gc.collect()
        _empty_accelerator_cache()
        _return_freed_memory_to_os()
        logger.info("Đã giải phóng %s", self.name)

    def maybe_release(self) -> None:
        """Giải phóng nếu cấu hình yêu cầu và không có khối ``hold()`` nào đang giữ model."""
        with self._lock:
            should_release = self.release_after_use and self._holds == 0
        if should_release:
            self.release()

    def run(self, fn: Callable[[T], R]) -> R:
        """Chạy ``fn(model)``; model được nạp lười và (tuỳ cấu hình) giải phóng ngay sau khi xong."""
        model = self.get()
        try:
            return fn(model)
        finally:
            del model  # bỏ tham chiếu cục bộ trước khi giải phóng
            self.maybe_release()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Không giải phóng model trong khối ``with`` (model vẫn chỉ được nạp khi thực sự cần)."""
        with self._lock:
            self._holds += 1
        try:
            yield
        finally:
            with self._lock:
                self._holds -= 1
            self.maybe_release()
