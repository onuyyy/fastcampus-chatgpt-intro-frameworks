import os  # 운영체제 환경변수 접근 모듈
import openai  # OpenAI API 사용 모듈
from dotenv import load_dotenv  # .env 파일에서 환경변수 로드
from fastapi import FastAPI  # FastAPI 웹 서버 프레임워크
from fastapi.middleware.cors import CORSMiddleware  # CORS 설정용 미들웨어
from pydantic import BaseModel  # 데이터 모델 정의 및 검증

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 생성, debug 모드 활성화
app = FastAPI(debug=True)

# CORS 설정: 모든 도메인(*)에서 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    message: str  # 사용자 메시지
    temperature: float = 1  # LLM 생성 텍스트 창의성 정도

# 사용자 정보 요청 함수 (실제 API 호출 대신 샘플 데이터 반환)
def request_user_info():
    # import requests
    # requests.get("https://api.xxx.com/users/username/info")
    return """
    - Like Asia food
    - Like to travel to Spain.
    - 30 years old.
    """

# 여행 계획 매뉴얼 요청 함수 (실제 API 호출 대신 샘플 데이터 반환)
def request_planning_manual():
    return """
    - 30 years old man likes eating food.
    - 30 years old man likes walking.
    """

# 시스템 메시지 정의 (LLM 역할 + 사용자 정보 + 계획 매뉴얼 포함)
SYSTEM_MSG = f"""You are a helpful travel assistant, Your name is Jini, 27 years old

Current User:
{request_user_info()}

Planning Manual:
{request_planning_manual()}
"""

# 사용자 메시지 기반 의도(intent) 분류 함수
def classify_intent(msg):
    prompt = f"""Your job is to classify intent.

    Choose one of the following intents:
    - travel_plan
    - customer_support
    - reservation

    User: {msg}
    Intent:
    """
    # GPT-4를 사용하여 의도 분류
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    # 의도 텍스트 반환
    return response.choices[0].message.content.strip()

# POST /chat 엔드포인트 정의
@app.post("/chat")
def chat(req: ChatRequest):
    # 사용자 메시지 의도 분류
    intent = classify_intent(req.message)

    # 여행 계획 요청일 경우 GPT-4로 답변 생성
    if intent == "travel_plan":
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},  # 시스템 메시지
                {"role": "user", "content": req.message},   # 사용자 메시지
            ],
            temperature=req.temperature,
        )
        return {"message": response.choices[0].message.content}

    # 고객 지원 요청일 경우 고정 메시지 반환
    elif intent == "customer_support":
        return {"message": "Here is customer support number: 1234567890"}

    # 예약 관련 요청일 경우 고정 메시지 반환
    elif intent == "reservation":
        return {"message": "Here is reservation number: 0987654321"}

# 로컬 서버 실행 (uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
