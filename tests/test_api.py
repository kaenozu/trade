import asyncio
import types

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

