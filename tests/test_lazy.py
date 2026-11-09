"""Unit test cho cơ chế nạp lười / giải phóng model (src/lazy.py)."""

from __future__ import annotations

import gc
import weakref

from src.lazy import LazyModel, current_rss_mb


class FakeModel:
    def predict(self, x: int) -> int:
        return x * 2


def make_lazy(release_after_use: bool) -> tuple[LazyModel[FakeModel], list[weakref.ref[FakeModel]]]:
    created: list[weakref.ref[FakeModel]] = []

    def loader() -> FakeModel:
        model = FakeModel()
        created.append(weakref.ref(model))
        return model

    return LazyModel("fake", loader, release_after_use=release_after_use), created


def test_model_is_not_loaded_until_first_use() -> None:
    lazy, created = make_lazy(release_after_use=False)
    assert not lazy.is_loaded and created == []
    assert lazy.run(lambda m: m.predict(2)) == 4
    assert lazy.is_loaded and lazy.load_count == 1


def test_model_kept_when_release_after_use_is_false() -> None:
    lazy, _ = make_lazy(release_after_use=False)
    lazy.run(lambda m: m.predict(1))
    lazy.run(lambda m: m.predict(1))
    assert lazy.is_loaded and lazy.load_count == 1


def test_model_released_and_garbage_collected_after_use() -> None:
    lazy, created = make_lazy(release_after_use=True)
    lazy.run(lambda m: m.predict(1))
    gc.collect()
    assert not lazy.is_loaded
    assert created[0]() is None, "model phải được thu hồi bộ nhớ sau khi giải phóng"
    lazy.run(lambda m: m.predict(1))
    assert lazy.load_count == 2


def test_hold_keeps_model_for_whole_batch_then_releases() -> None:
    lazy, created = make_lazy(release_after_use=True)
    with lazy.hold():
        assert not lazy.is_loaded, "hold() không được nạp model sớm"
        for x in range(5):
            lazy.run(lambda m, x=x: m.predict(x))
        assert lazy.is_loaded and lazy.load_count == 1
    gc.collect()
    assert not lazy.is_loaded and created[0]() is None


def test_nested_holds_release_only_at_outermost_exit() -> None:
    lazy, _ = make_lazy(release_after_use=True)
    with lazy.hold():
        with lazy.hold():
            lazy.get()
        assert lazy.is_loaded
    assert not lazy.is_loaded


def test_current_rss_mb_is_positive_on_linux() -> None:
    rss = current_rss_mb()
    assert rss != rss or rss > 0  # nan ngoài Linux, ngược lại phải dương
