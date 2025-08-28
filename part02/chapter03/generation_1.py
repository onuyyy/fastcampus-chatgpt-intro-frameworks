import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# .env 파일에 저장된 환경 변수 불러오기 (API 키 등)
load_dotenv()

# OpenAI API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션 생성 (디버그 모드 활성화)
app = FastAPI(debug=True)

# CORS 설정: 모든 도메인(*)에서 API 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 모든 출처 허용
    allow_credentials=True,       # 쿠키 포함한 요청 허용
    allow_methods=["*"],          # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],          # 모든 헤더 허용
)

# 요청 바디 모델 정의 (유저가 보내는 데이터 구조)
class ChatRequest(BaseModel):
    message: str                  # 유저 입력 메시지
    temperature: float = 1        # OpenAI temperature 값 (기본값 1)

# 시스템 프롬프트 (AI의 캐릭터/역할 정의)
SYSTEM_MSG = "You are a helpful travel assistant, Your name is Jini, 27 years old"

# /chat 엔드포인트 정의 (POST 요청)
@app.post("/chat")
def chat(req: ChatRequest):
    # OpenAI ChatCompletion API 호출
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",     # 사용할 모델 지정
        messages=[
            {"role": "system", "content": SYSTEM_MSG},  # AI의 역할 설정
            {"role": "user", "content": req.message},   # 유저 입력
        ],
        temperature=req.temperature, # 창의성 조절 값
    )
    # AI의 응답 반환
    return {"message": response.choices[0].message.content}

# 메인 실행 부분 (uvicorn으로 서버 실행)
if __name__ == "__main__":
    import uvicorn
    # 서버 실행 (0.0.0.0:8000 → 외부 접속 허용)
    uvicorn.run(app, host="0.0.0.0", port=8000)



# FastAPI : 파이썬 백엔드 프레임웤,
# HTTP 요청/응답 처리, 라우팅, 데이터 검증, 비동기 지원
# 자체적으로 WAS 를 포함하진 않지만 uvicorn(ASGI 서버) 위에서 실행