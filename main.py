# main.py
import numpy as np
import os
import uvicorn
import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from typing import List

from app.utils import clean_json_sequence, trim_zero_padding, sliding_window_gesture_detection
from app.model_loader import load_models
from app.gpt_router import router as gpt_router
from app.converter import convert_gestures_to_sentence
from app.tts import generate_tts


app = FastAPI()

# 정적 파일 제공
app.mount("/tts_audios", StaticFiles(directory="tts_audios"), name="tts")

# 루트 리다이렉션
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini 변환 라우터 포함
app.include_router(gpt_router)

# 모델 초기 로딩
encoder_model, gesture_hmms, ergodic_model = load_models()


# 요청 및 응답 스키마
class SequenceRequest(BaseModel):
    sequence: List[List[float]]

class Interval(BaseModel):
    start: int
    end: int
    label: str

class GestureResponse(BaseModel):
    intervals: List[Interval]

class TranslateResponse(BaseModel):
    sentence: str
    audio_base64: str

class TTSRequest(BaseModel):
    sentence: str


# 제스처 시퀀스를 제스처 라벨로 변환
@app.post("/predict_gesture", response_model=GestureResponse)
async def predict_gesture(req: SequenceRequest):
    try:
        expected_length = encoder_model.input_shape[2]
        sequence = clean_json_sequence(req.sequence, expected_length=expected_length)
        trimmed = trim_zero_padding(sequence)

        intervals = sliding_window_gesture_detection(
            continuous_sequence=trimmed,
            encoder_model=encoder_model,
            gesture_hmms=gesture_hmms,
            final_model=ergodic_model,
            window_size=20,
            step=2,
            threshold_diff=-300.0,
            min_merge_gap=3,
            min_interval_length=3
        )

        return GestureResponse(
            intervals=[Interval(start=int(s), end=int(e), label=l) for s, e, l in intervals]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 제스처 → 문장 변환
@app.post(
    "/predict_gesture_and_sentence",
    summary="제스처 시퀀스를 자연어 문장으로 변환",
    tags=["Gesture → Sentence"]
)
async def predict_gesture_and_sentence(req: SequenceRequest):
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

        gestures = [label for _, _, label in intervals] or ["none"]
        sentence = convert_gestures_to_sentence(gestures)

        return {"sentence": sentence}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 문장 → TTS 변환
@app.post(
    "/generate_tts",
    response_model=TranslateResponse,
    summary="자연어 문장을 TTS 음성(base64)으로 변환",
    tags=["TTS"]
)
async def generate_tts_audio(data: TTSRequest):
    try:
        sentence = data.sentence
        audio_path = generate_tts(sentence)

        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "sentence": sentence,
            "audio_base64": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
