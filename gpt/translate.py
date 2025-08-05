from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)
app = FastAPI()

class TranslateRequest(BaseModel):
    sentence: str

@app.post("/gpt/translate")
def translate(request:TranslateRequest):

    data_schema = {
        "type": "object",
        "properties": {
            "japanese": {
                "type" : "string",
                "description" : "일본어 번역"
            },
            "chinese": {
                "type" : "string",
                "description" : "중국어 번역"
            },
            "english": {
                "type" : "string",
                "description" : "영어 번역"
            }
        },
        "required": ["japanese", "chinese", "english"]
    }

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role" : "user",
                "content" : f""" "{request.sentence}"를 일본어, 영어, 중국어로 번역해줘."""
            }
        ],
        functions = [
            {
                "name" : "translate_text",
                "description": "문장을 일본어, 영어, 중국어로 번역",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "translate_text" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)
    return result