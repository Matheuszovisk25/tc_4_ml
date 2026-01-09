import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="LSTM (Notebook Style) • API + yfinance", layout="wide")

st.title("LSTM de Séries Temporais (estilo notebook) — via API + yfinance")
st.caption(
    "Mantém a sequência do notebook (títulos, explicações e gráficos), "
    "usando a API nos endpoints `POST /predict/yfinance` (projeção) e `POST /backtest/yfinance` (lado a lado no teste)."
)


# ============================================================
# Helpers
# ============================================================
@st.cache_data(ttl=60 * 30)
def yf_close_df(ticker: str, start: str, end: str, interval: str = "1d", auto_adjust: bool = False) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) + pd.Timedelta(days=1)

    df = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df["Close"]
        if hasattr(close, "columns") and len(close.columns) > 0:
            close = close.iloc[:, 0]
        out = pd.DataFrame({"Close": close})
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        out = df[["Close"]].copy()

    out = out.dropna()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.reset_index().rename(columns={"index": "Date"})
    return out


def api_health(base_url: str, timeout_s: int = 15) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    r = requests.get(url, timeout=timeout_s)
    try:
        data = r.json()
    except Exception:
        data = {"detail": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {data}")
    return data


def api_predict_yfinance(base_url: str, payload: Dict[str, Any], timeout_s: int = 120) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/predict/yfinance"
    r = requests.post(url, json=payload, timeout=timeout_s)
    try:
        data = r.json()
    except Exception:
        data = {"detail": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {data}")
    return data


def api_backtest_yfinance(base_url: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/backtest/yfinance"
    r = requests.post(url, json=payload, timeout=timeout_s)
    try:
        data = r.json()
    except Exception:
        data = {"detail": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {data}")
    return data


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)
    eps = 1e-9

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape}


def plot_history(df: pd.DataFrame, title: str, step: int = 200):
    fig = plt.figure(figsize=(18, 9))
    plt.plot(df["Close"])
    if df.shape[0] > step:
        ticks = list(range(0, df.shape[0], step))
    else:
        ticks = list(range(0, df.shape[0], max(1, df.shape[0] // 5)))
    plt.xticks(ticks, df["Date"].iloc[ticks].dt.strftime("%Y-%m-%d"), rotation=45)
    plt.xlabel("Datas", fontsize=18)
    plt.ylabel("Preço Médio", fontsize=18)
    plt.title(title, fontsize=30)
    plt.tight_layout()
    return fig


def plot_real_vs_pred(real: np.ndarray, pred: np.ndarray, dates: List[str], title: str, step: int = 20):
    fig = plt.figure(figsize=(18, 9))
    plt.plot(real, color="green", label="real")
    plt.plot(pred, color="red", label="previsão")
    if len(real) > step:
        ticks = list(range(0, len(real), step))
    else:
        ticks = list(range(0, len(real), max(1, len(real) // 5)))

    if dates:
        date_labels = [dates[i] for i in ticks]
    else:
        date_labels = ticks

    plt.xticks(ticks, date_labels, rotation=45)
    plt.xlabel("Datas", fontsize=18)
    plt.ylabel("Preço Médio", fontsize=18)
    plt.title(title, fontsize=30)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_future_projection(history_df: pd.DataFrame, pred_dates: List[str], preds: List[float], title: str, tail_points: int = 120):
    fig = plt.figure(figsize=(18, 9))

    real = history_df["Close"].tail(tail_points).to_numpy()
    real_dates = history_df["Date"].tail(tail_points).dt.strftime("%Y-%m-%d").tolist()

    plt.plot(real, color="green", label="real")
    if preds:
        plt.plot(range(len(real), len(real) + len(preds)), preds, color="red", label="previsão")

    combined_dates = real_dates + (pred_dates or [])
    ticks = list(range(0, len(combined_dates), max(1, len(combined_dates) // 8)))
    plt.xticks(ticks, [combined_dates[i] for i in ticks], rotation=45)

    plt.xlabel("Datas", fontsize=18)
    plt.ylabel("Preço Médio", fontsize=18)
    plt.title(title, fontsize=30)
    plt.legend()
    plt.tight_layout()
    return fig


# ============================================================
# Sidebar controls (parâmetros)
# ============================================================
with st.sidebar:
    st.header("Parâmetros (igual a ideia do notebook)")

    base_url = st.text_input("URL base da API", value="https://tc-4-ml.onrender.com")
    st.caption("A API precisa estar rodando com `/predict/yfinance` e `/backtest/yfinance` disponíveis.")

    st.divider()
    ticker = st.text_input("Ticker", value="WEG3.SA")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Data inicial", value=dt.date(2018, 1, 1))
    with c2:
        end_date = st.date_input("Data final", value=dt.date(2024, 12, 31))

    horizon = st.slider("Dias previstos (predicted_days)", 1, 60, 15, 1)
    interval = st.selectbox("Intervalo", ["1d", "1h"], index=0)
    auto_adjust = st.toggle("Auto-adjust (ajustado)", value=False)

    if interval == "1h":
        st.warning("`1h` gera MUITOS pontos e pode ficar pesado. Para ficar rápido, prefira `1d`.")

    st.divider()
    run = st.button("Notebook", type="primary")


# ============================================================
# Pré-checagem
# ============================================================
if not run:
    st.markdown(
        """
### Como usar
1) Suba a API (FastAPI)  
2) Ajuste **ticker**, **datas** e **predicted_days**  
3) Clique em **Rodar**

O app monta a página na mesma sequência do notebook:
- Carregamento
- Checagem de NaNs
- Visualização
- Separação treino/teste
- Normalização (conceito)
- Pré-processamento (conceito)
- Treinamento (conceito)
- Avaliação (real vs previsão no teste, lado a lado via `/backtest/yfinance`)
- Experimento (projeção futura via `/predict/yfinance`)
"""
    )
    st.stop()

if end_date < start_date:
    st.error("A data final precisa ser maior ou igual à data inicial.")
    st.stop()

payload = {
    "ticker": ticker.strip(),
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "horizon": int(horizon),
    "interval": interval,
    "auto_adjust": bool(auto_adjust),
}

# ============================================================
# Rodar: baixar histórico e chamar API
# ============================================================
with st.spinner("Baixando histórico (yfinance) e chamando API..."):
    df = yf_close_df(payload["ticker"], payload["start_date"], payload["end_date"], interval=interval, auto_adjust=auto_adjust)
    try:
        health = api_health(base_url)
        api_future = api_predict_yfinance(base_url, payload)       
        api_bt = api_backtest_yfinance(base_url, payload)          
    except Exception as e:
        st.error(f"Falha ao chamar a API: {e}")
        st.stop()

if df.empty:
    st.error("")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
prices = df["Close"].copy()

# ============================================================
# (0) Status
# ============================================================
st.subheader("Status da API")

st.divider()

# ============================================================
# 1) Carregamento dos Dados
# ============================================================
st.markdown("# Carregamento dos Dados")

st.code(
    "import pandas as pd\n"
    "import yfinance as yf\n"
    "df = yf.download(TICKER, start=..., end=...)\n",
    language="python",
)

st.write("**Tail:**")
st.dataframe(df.tail(), use_container_width=True)
st.write("**Head:**")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ============================================================
# 2) Tem linha do dataframe com dados faltando?
# ============================================================
st.markdown("# Tem linha do dataframe com dados faltando?")

nan_rows = df[df.isna().any(axis=1)]
st.write("Linhas com NaN (se existirem):")
st.dataframe(nan_rows, use_container_width=True)

st.write(f"Quantidade de linhas (antes de dropna): **{len(df)}**")
df_no_na = df.dropna().copy()
st.write(f"Quantidade de linhas (depois de dropna): **{len(df_no_na)}**")
st.caption("No seu notebook você fazia `df = df.dropna()`.")

st.divider()

# ============================================================
# 3) Visualização dos Dados
# ============================================================
st.markdown("# Visualização dos Dados")

fig1 = plot_history(df, title=f"Histórico de Preço {ticker}", step=200)
st.pyplot(fig1, clear_figure=True)

st.divider()

# ============================================================
# 4) Separação Teste e Treino
# ============================================================
st.markdown("# Separação Teste e Treino")

days_time_step = int(health.get("window_size") or api_future.get("window_size") or 10)
st.write(f"**days_time_step (window_size):** `{days_time_step}` (pego da API)")

training_size = int(len(prices) * 0.80)
test_size = len(prices) - training_size
train_data = np.array(prices[:training_size])
test_data = np.array(prices[training_size:])
input_data = np.array(prices[training_size - days_time_step:])

st.write(f"training_size: **{training_size}** | test_size: **{test_size}**")
st.write(f"train_data.shape: **{train_data.shape}** | test_data.shape: **{test_data.shape}**")

st.code(
    "training_size = int(len(prices) * 0.80)\n"
    "test_size = len(prices) - training_size\n"
    "train_data, input_data = np.array(prices[0:training_size]), np.array(prices[training_size - days_time_step:])\n"
    "test_data = np.array(prices[training_size:])\n",
    language="python",
)

fig = plt.figure(figsize=(18, 9))
plt.plot(df["Close"].iloc[:training_size], color="blue", label="treino")
plt.plot(df["Close"].iloc[training_size:], color="red", label="teste")
plt.xticks(range(0, df.shape[0], 200), df["Date"].iloc[::200].dt.strftime("%Y-%m-%d"), rotation=45)
plt.xlabel("Datas", fontsize=18)
plt.ylabel("Preço Médio", fontsize=18)
plt.title(f"Histórico de Preço {ticker}", fontsize=30)
plt.legend()
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

st.divider()

# ============================================================
# 5) Normalização dos Dados
# ============================================================
st.markdown("# Normalização dos Dados")

st.code(
    "from sklearn.preprocessing import MinMaxScaler\n"
    "scaler = MinMaxScaler(feature_range=(0,1))\n"
    "train_data_norm = scaler.fit_transform(train_data.reshape(-1,1))\n"
    "test_data_norm  = scaler.transform(input_data.reshape(-1,1))\n"
    "val_data_norm   = scaler.transform(test_data.reshape(-1,1))\n",
    language="python",
)

st.divider()

# ============================================================
# 6) Pré-processamento (Gerando X e y)
# ============================================================
st.markdown("# Pré-processamento (Gerando X e y)")

n_train_samples = max(0, len(train_data) - days_time_step)
n_test_samples = max(0, len(test_data))
st.write(f"X_train ≈ ({n_train_samples}, {days_time_step}, 1) | y_train ≈ ({n_train_samples}, 1)")
st.write(f"X_test ≈ ({n_test_samples}, {days_time_step}, 1)")

st.code(
    "#treino\n"
    "X_train, y_train = [], []\n"
    "for i in range(days_time_step, len(train_data)):\n"
    "    X_train.append(train_data_norm[i-days_time_step:i])\n"
    "    y_train.append(train_data_norm[i])\n"
    "\n"
    "#teste\n"
    "X_test = []\n"
    "for i in range(days_time_step, days_time_step + len(test_data)):\n"
    "    X_test.append(test_data_norm[i-days_time_step:i])\n"
    "\n"
    "#val\n"
    "X_val, y_val = [], []\n"
    "for i in range(days_time_step, len(test_data)):\n"
    "    X_val.append(val_data_norm[i-days_time_step:i])\n"
    "    y_val.append(val_data_norm[i])\n",
    language="python",
)

st.divider()

# ============================================================
# 7) Treinamento de Rede Neural
# ============================================================
st.markdown("# Treinamento de Rede Neural")

st.code(
    "from keras.models import Sequential\n"
    "from keras.layers import Dense, LSTM\n"
    "\n"
    "model = Sequential()\n"
    "model.add(LSTM(100, return_sequences=True, input_shape=(days_time_step, 1)))\n"
    "model.add(LSTM(100, return_sequences=False, input_shape=(days_time_step, 1)))\n"
    "model.add(Dense(1))\n"
    "model.compile(loss='mse', optimizer='adam')\n"
    "model.summary()\n",
    language="python",
)

st.code("h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)\n", language="python")

history_path = Path("history.json")  # mesmo diretório do app.py
if history_path.exists():
    import json
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    fig_loss = plt.figure(figsize=(18, 9))
    plt.plot(loss, label="loss")
    if val_loss:
        plt.plot(val_loss, label="val_loss")
    plt.title("Loss / Val Loss", fontsize=30)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig_loss, clear_figure=True)
else:
    st.warning("")

st.divider()

# ============================================================
# 8) Avaliação de Resultados (lado a lado)  
# ============================================================
st.markdown("# Avaliação de Resultados")

bt_dates = api_bt.get("dates") or []
bt_true = api_bt.get("y_true") or []
bt_pred = api_bt.get("y_pred") or []
bt_metrics = api_bt.get("metrics") or None

if not bt_dates or not bt_true or not bt_pred:
    st.warning("A API não retornou backtest completo. Veja o JSON bruto do backtest abaixo.")
    with st.expander("JSON bruto do backtest"):
        st.json(api_bt)
else:
    fig_bt = plot_real_vs_pred(
        real=np.array(bt_true, dtype=float),
        pred=np.array(bt_pred, dtype=float),
        dates=bt_dates,
        title=f"Projeção de Preço {ticker}",
        step=20,
    )
    st.pyplot(fig_bt, clear_figure=True)

    if not bt_metrics:
        bt_metrics = compute_metrics(np.array(bt_true), np.array(bt_pred))
        bt_metrics = {
            "mse": bt_metrics["MSE"],
            "rmse": bt_metrics["RMSE"],
            "mae": bt_metrics["MAE"],
            "mape": bt_metrics["MAPE_%"],
        }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MSE", f"{bt_metrics.get('mse', 0.0):.6f}")
    m2.metric("RMSE", f"{bt_metrics.get('rmse', 0.0):.6f}")
    m3.metric("MAE", f"{bt_metrics.get('mae', 0.0):.6f}")
    m4.metric("MAPE %", f"{bt_metrics.get('mape', 0.0):.3f}")

st.divider()

# ============================================================
# 9) (Projeção futura) 
# ============================================================
st.markdown("# Estimando sem informar o conjunto de teste todo")
st.write("A API faz a previsão iterativa usando a última janela do período escolhido (endpoint `/predict/yfinance`).")

pred_dates = api_future.get("predicted_dates") or []
preds = api_future.get("predictions") or []

if not pred_dates or not preds:
    st.warning("")
    with st.expander("JSON bruto da projeção (predict/yfinance)"):
        st.json(api_future)
else:
    fig3 = plot_future_projection(
        df,
        pred_dates=pred_dates,
        preds=preds,
        title=f"Projeção de Preço {ticker}",
        tail_points=120,
    )
    st.pyplot(fig3, clear_figure=True)

    pred_df = pd.DataFrame({"Date": pred_dates, "Prediction": preds})
    st.subheader("Previsões")
    st.dataframe(pred_df, use_container_width=True)

    st.download_button(
        "Baixar previsões (CSV)",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name=f"pred_{ticker}_{payload['start_date']}_{payload['end_date']}_h{horizon}.csv",
        mime="text/csv",
    )

st.divider()


# ============================================================
# 11) Referências
# ============================================================
st.markdown("# Referências")




