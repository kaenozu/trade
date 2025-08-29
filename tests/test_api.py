import pytest
from httpx import AsyncClient

from app.main import app
from app.services import data as data_service
from .utils import make_synthetic_ohlcv


@pytest.mark.asyncio
async def test_predict_endpoint_monkeypatched(monkeypatch):
    async def _run():
        def fake_fetch(ticker: str, period_days: int = 400, end=None):
            return make_synthetic_ohlcv(500)

        monkeypatch.setattr(data_service, "fetch_ohlcv", fake_fetch)
        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.post("/predict", json={"ticker": "TEST.T", "horizon_days": 7, "lookback_days": 400})
            assert resp.status_code == 200
            data = resp.json()
            assert data["ticker"] == "TEST.T"
            assert data["horizon_days"] == 7
            assert len(data["predictions"]) == 7
            tp = data["trade_plan"]
            assert set(tp.keys()) == {"buy_date", "sell_date", "confidence", "rationale"}

    await _run()


@pytest.mark.asyncio
async def test_quote_endpoint_success(monkeypatch):
    async def _run():
        def fake_fetch_direct(ticker: str, timeout: float = 5.0):
            return 1234.5, "2023-10-26"

        def fake_fetch_ohlcv(ticker: str, period_days: int = 400, end=None, ttl_seconds: int = 8 * 3600):
            # This should not be called in success case, but provide a fallback
            return make_synthetic_ohlcv(10)

        monkeypatch.setattr(data_service, "fetch_last_close_direct", fake_fetch_direct)
        monkeypatch.setattr(data_service, "fetch_ohlcv", fake_fetch_ohlcv)

        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/quote?ticker=7203.T")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ticker"] == "7203.T"
            assert data["price"] == 1234.5
            assert data["asof"] == "2023-10-26"

    await _run()


@pytest.mark.asyncio
async def test_quote_endpoint_invalid_ticker(monkeypatch):
    async def _run():
        # Mock to ensure it raises an exception for invalid ticker
        def fake_fetch_direct(ticker: str, timeout: float = 5.0):
            raise ValueError("Invalid ticker")

        def fake_fetch_ohlcv(ticker: str, period_days: int = 400, end=None, ttl_seconds: int = 8 * 3600):
            raise ValueError("Invalid ticker")

        monkeypatch.setattr(data_service, "fetch_last_close_direct", fake_fetch_direct)
        monkeypatch.setattr(data_service, "fetch_ohlcv", fake_fetch_ohlcv)

        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/quote?ticker=")
            assert resp.status_code == 400

    await _run()


@pytest.mark.asyncio
async def test_quotes_endpoint_success(monkeypatch):
    async def _run():
        def fake_fetch_direct(ticker: str, timeout: float = 5.0):
            if ticker == "7203.T":
                return 1234.5, "2023-10-26"
            elif ticker == "9984.T":
                return 6789.0, "2023-10-26"
            else:
                raise ValueError("Invalid ticker")

        def fake_fetch_ohlcv(ticker: str, period_days: int = 400, end=None, ttl_seconds: int = 8 * 3600):
            # This should not be called in success case, but provide a fallback
            return make_synthetic_ohlcv(10)

        monkeypatch.setattr(data_service, "fetch_last_close_direct", fake_fetch_direct)
        monkeypatch.setattr(data_service, "fetch_ohlcv", fake_fetch_ohlcv)

        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/quotes?tickers=7203.T,9984.T")
            assert resp.status_code == 200
            data = resp.json()
            assert "quotes" in data
            quotes = data["quotes"]
            assert len(quotes) == 2
            # Check if the tickers are in the response, regardless of order
            tickers_in_response = [q["ticker"] for q in quotes]
            assert "7203.T" in tickers_in_response
            assert "9984.T" in tickers_in_response
            # Check one of the quotes for correct data
            quote_7203 = next((q for q in quotes if q["ticker"] == "7203.T"), None)
            assert quote_7203 is not None
            assert quote_7203["price"] == 1234.5
            assert quote_7203["asof"] == "2023-10-26"

    await _run()


@pytest.mark.asyncio
async def test_quotes_endpoint_empty_tickers():
    async def _run():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/quotes?tickers=")
            assert resp.status_code == 400

    await _run()

