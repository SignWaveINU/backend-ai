# app/gpt_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.converter import convert_gestures_to_sentence

router = APIRouter()

class GestureList(BaseModel):
    gestures: List[str]

@router.post("/convert_gestures", summary="제스처를 자연어 문장으로 변환", tags=["GPT Conversion"])
async def convert_gestures(req: GestureList):
    sentence = convert_gestures_to_sentence(req.gestures)
    return {"sentence": sentence}
