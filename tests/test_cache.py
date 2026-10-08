from src.tools.cache import ttl_cache


def test_second_call_is_cached(tmp_path):
    calls = []

    @ttl_cache("test", ttl=60, cache_dir=tmp_path)
    def fetch(x):
        calls.append(x)
        return {"x": x}

    assert fetch(1) == {"x": 1}
    assert fetch(1) == {"x": 1}
    assert calls == [1]


def test_expired_entry_refetches(tmp_path):
    calls = []

    @ttl_cache("test", ttl=0, cache_dir=tmp_path)
    def fetch(x):
        calls.append(x)
        return x

    fetch(1)
    fetch(1)
    assert calls == [1, 1]


def test_none_is_not_cached(tmp_path):
    calls = []

    @ttl_cache("test", ttl=60, cache_dir=tmp_path)
    def fetch():
        calls.append(1)
        return None

    fetch()
    fetch()
    assert len(calls) == 2
