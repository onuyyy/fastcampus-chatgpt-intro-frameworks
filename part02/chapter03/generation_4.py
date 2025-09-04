import os
from dotenv import load_dotenv

from langchain.chains import LLMMathChain
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

# .env에서 API 키 불러오기
load_dotenv()

# ----------------------------
# LLM 및 도구 설정
# ----------------------------
llm = OpenAI(temperature=0)

# 수학 계산용 체인
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# SerpAPI 검색 도구
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

# Tools 목록 정의
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math"
    ),
]

# ----------------------------
# Agent 생성
# ----------------------------
# ChatOpenAI 모델 설정
model = ChatOpenAI(temperature=0)

# Agent 초기화
agent = initialize_agent(
    tools,
    model,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ----------------------------
# Agent 실행
# ----------------------------
response = agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)

print(response)
