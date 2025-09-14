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
                "description" : "course description in one line"
            },
            "long_description": {
                "type" : "string",
                "description" : "course description about three lines long"
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
                        "store name" : {request.name},
                        "store address" : {request.address},
                        "store main category" : {request.mainCategory},
                        "store sub categories" : {request.subCategory},
                        "store strengths (what the owner thinks)" : {request.strength},
                        "special features recommended by the owner" : {request.recommendation}
                    }}" Based on this data, generate a one-line and three-lines description of the store in Korean.
                      Add a bit more explanation, but do not make up any false information.   """
            }
        ],
        functions = [
            {
                "name" : "get_store_description",
                "description": "Retrieve store descriptions",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "get_store_description" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)
    return result