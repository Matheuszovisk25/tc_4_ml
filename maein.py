import os
import json
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from joblib import load as joblib_load
from keras.models import load_model
from numpy.lib.stride_tricks import sliding_window_view
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .schemas import (
    PredictRequest,
    PredictResponse,
    YFinancePredictRequest,
    YFinancePredictResponse,
    BacktestResponse,
)

REQUEST_COUNT = Counter("api_requests_total", "Total de requests na API", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latência por endpoint (s)", ["endpoint"])
INFER_LATENCY = Histogram("model_inference_latency_seconds", "Latência de inferência do modelo (s)")

# Cache simples para yfinance (em memória)
_YF_CACHE = {}  # key -> (ts, close_values, dates, last_obs_date)
_YF_TTL = int(os.getenv("YF_CACHE_TTL", str(60 * 30)))  # 30 min padrão

# Performance do backtest/predict
MAX_BACKTEST_POINTS = int(os.getenv("MAX_BACKTEST_POINTS", "600"))   # limita pontos retornados/plotados
PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "1024"))    # batch_size do predict em batch


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _validate_date(s: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(s, format="%Y-%m-%d", errors="raise")
        return ts
    except Exception:
        raise HTTPException(status_code=422, detail=f"Data inválida '{s}'. Use YYYY-MM-DD.")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)
    eps = 1e-9
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


load_dotenv()

MODEL_PATH = _get_env("MODEL_PATH", "models/lstm_model.keras")
SCALER_PATH = _get_env("SCALER_PATH", "models/scaler.pkl")
METADATA_PATH = _get_env("METADATA_PATH", "models/metadata.json")

app = FastAPI(
    title="Stock LSTM Prediction API (com yfinance)",
    version="1.3.0",
    description="API para inferência de um modelo LSTM (séries temporais) e busca de histórico via yfinance.",
)

model = None
scaler = None
metadata = None
window_size = None


@app.on_event("startup")
def startup():
    global model, scaler, metadata, window_size

    metadata = _read_json(METADATA_PATH) or {}
    window_size = int(metadata.get("window_size", 0)) if metadata.get("window_size") else 0

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"MODEL_PATH não encontrado: {MODEL_PATH}. "
            "Salve seu modelo (model.save) antes de subir a API."
        )
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"SCALER_PATH não encontrado: {SCALER_PATH}. "
            "Salve seu scaler (joblib.dump) antes de subir a API."
        )

    model = load_model(MODEL_PATH)
    scaler = joblib_load(SCALER_PATH)

    try:
        inferred = int(model.input_shape[1])
        if window_size <= 0 or window_size != inferred:
            window_size = inferred
    except Exception:
        if window_size <= 0:
            window_size = 60


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "metadata_path": METADATA_PATH,
        "window_size": window_size,
        "yf_cache_ttl": _YF_TTL,
        "max_backtest_points": MAX_BACKTEST_POINTS,
        "predict_batch_size": PREDICT_BATCH_SIZE,
    }


def _predict_iterative_from_history(history: np.ndarray, horizon: int) -> np.ndarray:
    if history.ndim != 1:
        history = history.reshape(-1)

    hist_norm = scaler.transform(history.reshape(-1, 1)).reshape(-1)
    if hist_norm.shape[0] < window_size:
        raise ValueError("Histórico menor que window_size")

    window = hist_norm[-window_size:].reshape(1, window_size, 1)

    preds_norm: List[float] = []
    for _ in range(horizon):
        t0 = time.perf_counter()
        yhat = model.predict(window, verbose=0)
        INFER_LATENCY.observe(time.perf_counter() - t0)

        next_val = float(yhat.reshape(-1)[0])
        preds_norm.append(next_val)

        w = window.reshape(window_size)
        w = np.append(w[1:], next_val)
        window = w.reshape(1, window_size, 1)

    preds = scaler.inverse_transform(np.array(preds_norm).reshape(-1, 1)).reshape(-1)
    return preds


def _next_business_dates(last_date: pd.Timestamp, n: int) -> List[str]:
    rng = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n)
    return [d.strftime("%Y-%m-%d") for d in rng]


def _fetch_close_series(ticker: str, start_date: str, end_date: str, interval: str, auto_adjust: bool):
    start_ts = _validate_date(start_date)
    end_ts = _validate_date(end_date)
    if end_ts < start_ts:
        raise HTTPException(status_code=422, detail="end_date precisa ser >= start_date.")

    # Cache (mesmos parâmetros = mesmo retorno)
    key = (ticker, start_date, end_date, interval, bool(auto_adjust))
    now = time.time()
    cached = _YF_CACHE.get(key)
    if cached is not None:
        ts, close_values, dates, last_obs_date = cached
        if now - ts < _YF_TTL:
            return close_values, dates, last_obs_date

    df = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Sem dados para ticker='{ticker}' no período {start_date}..{end_date}.",
        )

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        else:
            raise HTTPException(status_code=500, detail="Não encontrei coluna 'Close' no retorno do yfinance.")
        if hasattr(close, "columns") and len(close.columns) > 0:
            close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise HTTPException(status_code=500, detail="Não encontrei coluna 'Close' no retorno do yfinance.")
        close = df["Close"]

    close = close.dropna()
    if close.empty:
        raise HTTPException(status_code=404, detail="Série de Close vazia após remover NaNs.")

    idx = pd.to_datetime(close.index)

    # remove timezone se existir
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass

    dates = idx.strftime("%Y-%m-%d").tolist()
    last_obs_date = pd.to_datetime(idx[-1])

    close_values = close.values.astype(float)

    # grava cache
    _YF_CACHE[key] = (now, close_values, dates, last_obs_date)

    return close_values, dates, last_obs_date


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    endpoint = "/predict"
    t0 = time.perf_counter()
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        hist = np.array(req.history, dtype=float)
        if hist.size < window_size:
            raise HTTPException(
                status_code=422,
                detail=f"history precisa ter pelo menos {window_size} pontos (recebido: {hist.size}).",
            )

        preds = _predict_iterative_from_history(hist, req.horizon)

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
        return PredictResponse(
            predictions=[float(x) for x in preds.tolist()],
            window_size=int(window_size),
            horizon=int(req.horizon),
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            metadata=metadata or None,
        )
    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)


@app.post("/predict/yfinance", response_model=YFinancePredictResponse)
def predict_yfinance(req: YFinancePredictRequest):
    endpoint = "/predict/yfinance"
    t0 = time.perf_counter()
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        close_values, _, last_obs_date = _fetch_close_series(
            req.ticker, req.start_date, req.end_date, req.interval, req.auto_adjust
        )

        if close_values.size < window_size:
            raise HTTPException(
                status_code=422,
                detail=f"Período retornou {close_values.size} pontos, mas window_size={window_size}. "
                       f"Aumente o intervalo de datas ou reduza a janela do modelo.",
            )

        preds = _predict_iterative_from_history(close_values, req.horizon)
        predicted_dates = _next_business_dates(last_obs_date, req.horizon)

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
        return YFinancePredictResponse(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            last_observation_date=last_obs_date.strftime("%Y-%m-%d"),
            predicted_dates=predicted_dates,
            n_observations_used=int(min(close_values.size, window_size)),
            predictions=[float(x) for x in preds.tolist()],
            window_size=int(window_size),
            horizon=int(req.horizon),
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            metadata=metadata or None,
        )
    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)


@app.post("/backtest/yfinance", response_model=BacktestResponse)
def backtest_yfinance(req: YFinancePredictRequest):
    endpoint = "/backtest/yfinance"
    t0 = time.perf_counter()
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        close_values, dates, _ = _fetch_close_series(
            req.ticker, req.start_date, req.end_date, req.interval, req.auto_adjust
        )

        n = close_values.size
        if n <= window_size + 5:
            raise HTTPException(status_code=422, detail="Poucos dados para backtest. Aumente o período.")
        split = int(n * 0.8)
        if split <= window_size:
            raise HTTPException(status_code=422, detail="Poucos dados: split <= window_size. Aumente o período.")

        # normaliza série inteira
        norm = scaler.transform(close_values.reshape(-1, 1)).reshape(-1)

        # janelas 1-step para toda a série
        X_all = sliding_window_view(norm, window_size)[:-1]      # (n-window_size, window_size)
        y_all = close_values[window_size:]                        # (n-window_size,)
        d_all = dates[window_size:]                               # (n-window_size,)

        # recorta o teste: alvo começa em split
        start_pos = split - window_size
        X_test = X_all[start_pos:]
        y_true = y_all[start_pos:]
        d_test = d_all[start_pos:]

        # limita quantidade de pontos (mais rápido + resposta menor)
        if MAX_BACKTEST_POINTS > 0 and X_test.shape[0] > MAX_BACKTEST_POINTS:
            X_test = X_test[-MAX_BACKTEST_POINTS:]
            y_true = y_true[-MAX_BACKTEST_POINTS:]
            d_test = d_test[-MAX_BACKTEST_POINTS:]

        # predict batch (rápido)
        X_test = X_test.reshape(-1, window_size, 1).astype(np.float32)

        t_inf = time.perf_counter()
        pred_norm = model.predict(
            X_test,
            verbose=0,
            batch_size=PREDICT_BATCH_SIZE,
        ).reshape(-1, 1)
        INFER_LATENCY.observe(time.perf_counter() - t_inf)

        y_pred = scaler.inverse_transform(pred_norm).reshape(-1)

        metrics = _compute_metrics(y_true, y_pred)

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
        return BacktestResponse(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            auto_adjust=req.auto_adjust,
            window_size=int(window_size),
            split_index=int(split),
            dates=list(d_test),
            y_true=[float(x) for x in y_true.tolist()],
            y_pred=[float(x) for x in y_pred.tolist()],
            metrics=metrics,
        )
    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
