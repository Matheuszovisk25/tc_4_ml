# Stock LSTM Prediction API (FastAPI) + yfinance

A API permite **duas formas** de prever:

1) Enviar o histórico manualmente: `POST /predict`
2) Buscar histórico do Yahoo Finance via `yfinance`: `POST /predict/yfinance`

## Rodar
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: `http://localhost:8000/docs`

## Endpoint yfinance
`POST /predict/yfinance`

Exemplo:
```json
{
  "ticker": "WEG3.SA",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "horizon": 15,
  "interval": "1d",
  "auto_adjust": false
}
```

A resposta inclui `predicted_dates` (próximos dias úteis) + `predictions`.

> Dica: garanta que o `window_size` no `metadata.json` (ou do input do modelo) bata com o seu treinamento.
