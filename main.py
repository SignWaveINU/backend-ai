# main.py
import numpy as np
import os
import uvicorn

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


# FastAPI 애플리케이션 생성
app = FastAPI()

# 정적 파일 제공: "/tts_audios/파일명" 경로로 접근 가능하도록 설정
app.mount("/tts_audios", StaticFiles(directory="tts_audios"), name="tts")

# 루트 접근 시 Swagger로 리다이렉트
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPT 변환 라우터 등록
app.include_router(gpt_router)

# 모델 로드 (최초 1회 실행)
encoder_model, gesture_hmms, ergodic_model = load_models()

# 요청/응답 스키마 정의
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
    audio_base64: str  # audio_url → audio_base64로 변경


# 제스처 예측 API
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

# 제스처 → 문장 + TTS 변환 API
@app.post(
    "/predict_gesture_and_translate",
    response_model=TranslateResponse,
    summary="제스처 시퀀스를 자연어 문장, TTS으로 변환",
    tags=["Gesture + Gemini"]
)
async def predict_gesture_and_translate(req: SequenceRequest):
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

        # TTS 음성 파일 생성
        audio_path = generate_tts(sentence)

        # 🔽 mp3 파일을 base64로 인코딩
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "sentence": sentence,
            "audio_base64": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
