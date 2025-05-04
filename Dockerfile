FROM python:3.9-slim

# 작업 디렉토리 생성
WORKDIR /app

# requirements 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# .env 포함 전체 프로젝트 복사
COPY . .

# 모델 디렉토리 명시적으로 생성 (필요시)
RUN mkdir -p /app/models

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

