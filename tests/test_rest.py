from decimal import Decimal

import pytest

from spreadbot.venues.rest import (
    MAX_ORDER_LIMIT,
    LighterApiError,
    LighterRateLimited,
    LighterRest,
    _aggregate,
)

D = Decimal


def test_aggregate_sums_orders_at_the_same_price():
    raw = [
        {"price": "100.0", "remaining_base_amount": "1.5"},
        {"price": "100.0", "remaining_base_amount": "2.5"},
        {"price": "99.0", "remaining_base_amount": "1.0"},
    ]
    assert _aggregate(raw) == [(D("100.0"), D("4.0")), (D("99.0"), D("1.0"))]


def test_aggregate_sorts_each_side_correctly():
    raw = [{"price": p, "remaining_base_amount": "1"} for p in ("101", "99", "100")]
    assert [p for p, _ in _aggregate(raw)] == [D(101), D(100), D(99)]
    assert [p for p, _ in _aggregate(raw, ascending=True)] == [D(99), D(100), D(101)]


def test_aggregate_drops_empty_and_malformed_levels():
    raw = [
        {"price": "100", "remaining_base_amount": "0"},
        {"price": "0", "remaining_base_amount": "5"},
        {"price": "101", "remaining_base_amount": "2"},
    ]
    assert _aggregate(raw) == [(D(101), D(2))]


class _FakeResponse:
    def __init__(self, status, text, headers=None):
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    closed = False

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0

    def get(self, url, params=None):
        self.calls += 1
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def _rest_with(*responses):
    rest = LighterRest("https://example.invalid")
    session = _FakeSession(*responses)
    rest._session = session

    async def _noop_start():
        return None

    rest.start = _noop_start
    return rest, session


async def test_rate_limit_is_retried_then_succeeds():
    rest, session = _rest_with(
        _FakeResponse(429, "", {"Retry-After": "0"}),
        _FakeResponse(200, '{"code":200,"ok":true}'),
    )
    assert (await rest._get("orderBookOrders"))["ok"] is True
    assert session.calls == 2


async def test_persistent_rate_limit_raises_its_own_error():
    rest, _ = _rest_with(_FakeResponse(429, "", {"Retry-After": "0"}))
    with pytest.raises(LighterRateLimited) as excinfo:
        await rest._get("orderBookOrders", retries=2)
    # The feed keys its whole backoff off this type, so it must not be a plain
    # LighterApiError with a 429 buried in the message.
    assert excinfo.value.code == 429


async def test_non_json_body_reports_the_status_not_a_parse_error():
    rest, _ = _rest_with(_FakeResponse(503, "<html>upstream unavailable</html>"))
    with pytest.raises(LighterApiError) as excinfo:
        await rest._get("orderBookOrders", retries=1)
    assert "503" in str(excinfo.value)
    assert "non-JSON" in str(excinfo.value)


async def test_empty_body_is_reported_clearly():
    rest, _ = _rest_with(_FakeResponse(200, ""))
    with pytest.raises(LighterApiError) as excinfo:
        await rest._get("orderBookOrders", retries=1)
    assert "non-JSON" in str(excinfo.value)


async def test_envelope_error_code_is_surfaced():
    rest, _ = _rest_with(_FakeResponse(200, '{"code":20001,"message":"invalid param "}'))
    with pytest.raises(LighterApiError) as excinfo:
        await rest._get("orderBookOrders", retries=1)
    assert "20001" in str(excinfo.value)
    assert "invalid param" in str(excinfo.value)


class _CapturingRest(LighterRest):
    """Records the params a call would have sent instead of hitting the network."""

    def __init__(self) -> None:
        super().__init__("https://example.invalid")
        self.params = None

    async def _get(self, path, params=None, *, retries=3):
        self.params = params
        return {"code": 200, "bids": [], "asks": []}


@pytest.mark.parametrize(
    "asked,sent",
    [(10, 10), (250, 250), (251, MAX_ORDER_LIMIT), (10_000, MAX_ORDER_LIMIT), (0, 1), (-5, 1)],
)
async def test_order_book_limit_is_clamped_to_what_the_api_accepts(asked, sent):
    # orderBookOrders answers anything above 250 with a bare "invalid param",
    # which would silently blank the REST feed rather than fail loudly.
    rest = _CapturingRest()
    await rest.order_book_levels(1, limit=asked)
    assert rest.params["limit"] == sent
