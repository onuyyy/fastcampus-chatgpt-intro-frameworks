import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you do?"},
    ],
    # 파라미터
    temperature=0.9,  # gpt 답변의 랜덤성 조절 
    n=2, # gpt 답변의 개수  
    # stop = [","], # 답변에 쉼표가 나오면 끝내라는 파라미터
    max_tokens=20 # 답변의 길이
)

# gpt 에게 온 답변을 출력해본다
for res in response.choices:
    print(res.message.content)  # <- 이렇게 속성 접근
