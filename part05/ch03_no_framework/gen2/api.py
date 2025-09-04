import os
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI  # 최신 SDK

load_dotenv()

# OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates/1_extract_idea.txt")
STEP2_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates/2_write_outline.txt")
STEP3_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates/3_write_plot.txt")
WRITE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates/6_write_chapter.txt")


class UserRequest(BaseModel):
    genre: str
    characters: List[Dict[str, str]]
    news_text: str


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def request_gpt_api(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    max_token: int = 500,
    temperature: float = 0.8,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_token,
        temperature=temperature,
    )
    return response.choices[0].message.content


@app.post("/writer")
def generate_novel(req: UserRequest) -> Dict[str, str]:
    context = {}

    # characters 리스트를 문자열로 변환
    #characters_text = "\n".join([f"{c['name']} - {c['role']}" for c in req.characters])

    # 1. 아이디어 뽑기
    novel_idea_prompt_template = read_prompt_template(STEP1_PROMPT_TEMPLATE)
    novel_idea_prompt = novel_idea_prompt_template.format(
        genre=req.genre,
        characters=req.characters,
        # characters=characters_text,
        news_text=req.news_text,
    )
    context["novel_idea"] = request_gpt_api(novel_idea_prompt)

    # 2. 아웃라인 작성
    novel_outline_prompt_template = read_prompt_template(STEP2_PROMPT_TEMPLATE)
    novel_outline_prompt = novel_outline_prompt_template.format(
        genre=req.genre,
        characters=req.characters,
        # characters=characters_text,
        news_text=req.news_text,
        novel_idea=context["novel_idea"],
    )
    context["novel_outline"] = request_gpt_api(novel_outline_prompt)

    # 3. 소설 플롯 작성
    novel_plot_prompt_template = read_prompt_template(STEP3_PROMPT_TEMPLATE)
    novel_plot_prompt = novel_plot_prompt_template.format(
        genre=req.genre,
        characters=req.characters,
        # characters=characters_text,
        news_text=req.news_text,
        novel_idea=context["novel_idea"],
        novel_outline=context["novel_outline"],
    )
    context["novel_plot"] = request_gpt_api(novel_plot_prompt)

    # 4. 소설 챕터 작성
    write_prompt_template = read_prompt_template(WRITE_PROMPT_TEMPLATE)
    context["novel_chapter"] = []
    for chapter_number in range(1, 3):
        write_prompt = write_prompt_template.format(
            genre=req.genre,
            characters=req.characters,
            # characters=characters_text,
            news_text=req.news_text,
            novel_idea=context["novel_idea"],
            novel_outline=context["novel_outline"],
            novel_plot=context["novel_plot"],
            chapter_number=chapter_number,
        )
        context["novel_chapter"].append(request_gpt_api(write_prompt))

    contents = "\n\n".join(context["novel_chapter"])
    return {"results": contents}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
