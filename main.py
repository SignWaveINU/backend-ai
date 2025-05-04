import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.utils import clean_json_sequence
from pydantic import BaseModel
from typing import List
import uvicorn

from app.utils import trim_zero_padding, sliding_window_gesture_detection
from app.model_loader import load_models

# 모델 로드 (최초 1회)
encoder_model, gesture_hmms, ergodic_model = load_models()

app = FastAPI()

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 스키마
class SequenceRequest(BaseModel):
    sequence: List[List[float]]

class Interval(BaseModel):
    start: int
    end: int
    label: str

class GestureResponse(BaseModel):
    intervals: List[Interval]

@app.post("/predict_gesture", response_model=GestureResponse)
async def predict_gesture(req: SequenceRequest):
    try:
        sequence = clean_json_sequence(req.sequence, expected_length=126)
        trimmed = trim_zero_padding(sequence)

        intervals = sliding_window_gesture_detection(
            continuous_sequence=trimmed,
            encoder_model=encoder_model,
            gesture_hmms=gesture_hmms,
            final_model=ergodic_model,
            window_size=20,
            step=2,
            threshold_diff=0.0,
            min_merge_gap=5,
            min_interval_length=5,
        )

        return GestureResponse(
            intervals=[Interval(start=int(s), end=int(e), label=l) for s, e, l in intervals]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
