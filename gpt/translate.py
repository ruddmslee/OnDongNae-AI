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

system_prompt = """
You are a localization assistant that translates Korean to Japanese, Chinese (Simplified), and English.

Rules:
1. Do not literally translate Korean dish names or proper nouns. Use Revised Romanization for Korean food / proper nouns by default (e.g., 냉면 -> Naengmyeon)
2. When unsure, prefer romanization over a guessed literal meaning.
3. Translate standard business terms into the target language. (e.g., 상회 -> store, 약국 -> pharmacy)
4. Your responses will be returned through the tool/function 'translate_text' with the schema {japanese, chinese, english}.
5. Always start each translated sentence with a capital letter.

"""

class TranslateRequest(BaseModel):
    sentence: str

@app.post("/")
def translate(request:TranslateRequest):

    data_schema = {
        "type": "object",
        "properties": {
            "japanese": {
                "type" : "string",
                "description" : "translate into japanese"
            },
            "chinese": {
                "type" : "string",
                "description" : "translate into chinese"
            },
            "english": {
                "type" : "string",
                "description" : "translate into english"
            }
        },
        "required": ["japanese", "chinese", "english"]
    }

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        temperature=0,
        messages = [
            {
                "role" : "system",
                "content" : system_prompt
            },
            {
                "role" : "user",
                "content" : f" Translate this into JA, EN, ZH-HANS: {request.sentence}"
            }
        ],
        functions = [
            {
                "name" : "translate_text",
                "description": "translate sentence into japanese, chinese, english",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "translate_text" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)
    return result