from pydantic import BaseModel, Field
from typing import List, Optional, Literal



class PredictRequest(BaseModel):
    history: List[float] = Field(
        ...,
        description="Série histórica (ex.: preços de fechamento). Precisa ter ao menos window_size pontos.",
        min_length=2,
    )
    horizon: int = Field(15, ge=1, le=365, description="Quantos passos/dias à frente prever.")


class YFinancePredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker do ativo. Ex.: WEG3.SA, PETR4.SA, AAPL.")
    start_date: str = Field(..., description="Data inicial (YYYY-MM-DD).")
    end_date: str = Field(..., description="Data final (YYYY-MM-DD).")
    horizon: int = Field(15, ge=1, le=365, description="Quantos passos/dias úteis à frente prever.")
    interval: Literal["1d", "1h"] = Field("1d", description="Intervalo do Yahoo Finance.")
    auto_adjust: bool = Field(False, description="Se True, baixa preços ajustados.")


class PredictResponse(BaseModel):
    predictions: List[float]
    window_size: int
    horizon: int
    model_path: str
    scaler_path: str
    metadata: Optional[dict] = None


class YFinancePredictResponse(PredictResponse):
    ticker: str
    start_date: str
    end_date: str
    last_observation_date: Optional[str] = None
    predicted_dates: Optional[List[str]] = None
    n_observations_used: int

from typing import List, Optional, Dict
from pydantic import BaseModel

class BacktestResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    interval: str
    auto_adjust: bool
    window_size: int
    split_index: int
    dates: List[str]          # datas do TESTE
    y_true: List[float]       # Close real no teste
    y_pred: List[float]       # previsão 1-step alinhada
    metrics: Optional[Dict[str, float]] = None