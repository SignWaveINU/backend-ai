#app/converter.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def convert_gestures_to_sentence(gestures: list[str]) -> str:
    prompt = f"""
너는 수어 인식 AI의 예측 결과를 받아 자연스러운 한국어 문장으로 변환해주는 역할이야.

입력은 다음 두 가지 형식 중 하나야:

1. 
{{{{ 
    "gesture": ["예측된 제스처 이름"] 
}}}}

2. 
{{{{ 
    "gesture": ["제스처1", "제스처2", ..., "제스처N"] 
}}}}

각 제스처는 일상적인 상황에서 사용할 수 있는 자연스러운 문장으로 바꿔줘.
예시:
- "피" → "피가 나요"
- "의사","상담" → "의사와 상담하고 싶어요"
- "전화번호" → "전화번호를 알려주세요"
- "약","먹다" → "약을 먹었어요"
- "병원" → "병원에 가고 싶어요"
- "보호자","부르다" → "보호자 불러 주세요"
- "아프다" → "아파요"
- "걱정되다" → "걱정돼요"

규칙:
- 입력이 제스처 하나일 경우 → 해당 제스처에 대한 문장 하나를 출력해.
- 입력이 제스처 여러개일 경우 → 각 제스처를 순서대로 자연스럽게 합친 문장 하나를 출력해.
- 제스처가 "none"이면 → "무슨 제스처인지 모르겠어요"로 출력해.
- 결과는 자연스러운 한국어 문장으로, 최대한 간결하고 공손하게 작성해.
- 출력은 오직 문장들만 해. 설명하거나 JSON으로 감싸지 마.

입력: {gestures}
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()