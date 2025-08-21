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

class StoreDescriptionRequest(BaseModel):
    name : str
    address : str
    mainCategory : str
    subCategory : list[str]
    strength : str
    recommendation : str

@app.post("/")
def get_store_description(request : StoreDescriptionRequest):
    data_schema = {
        "type": "object",
        "properties": {
            "short_description": {
                "type" : "string",
                "description" : "한 줄 설명"
            },
            "long_description": {
                "type" : "string",
                "description" : "세 줄 설명"
            }
        },
        "required": ["short_description", "long_description"]
    }

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role" : "user",
                "content" : f""" 
                    "{{
                        "가게 이름" : {request.name},
                        "가게 주소" : {request.address},
                        "가게 업종" : {request.mainCategory},
                        "가게 세부 업종" : {request.subCategory},
                        "사장님이 생각하는 가게의 장점" : {request.strength},
                        "사장님이 추천하고 싶은 가게의 특징" : {request.recommendation}
                    }} 이 데이터를 바탕으로 가게에 대한 한 줄 설명, 여러 줄 설명을 생성해줘. 조금 더 설명을 덧붙여줘. 거짓말은 하지 말아줘."""
            }
        ],
        functions = [
            {
                "name" : "get_store_description",
                "description": "가게 설명 불러오기",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "get_store_description" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)
    return result