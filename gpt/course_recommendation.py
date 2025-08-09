from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
import numpy as np
import faiss
from typing import Dict, Any

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)
app = FastAPI()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
FAISS_INDEX_DIR = os.path.join(desktop_path, 'database/faiss_index')
FAISS_INDEX_NAME = 'stores.index'
METADATA_NAME = 'metadata.json'

if not os.path.exists(FAISS_INDEX_DIR):
    os.makedirs(FAISS_INDEX_DIR)

index_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)
metadata_path = os.path.join(FAISS_INDEX_DIR, METADATA_NAME)


# 임베딩
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response =  client.embeddings.create(input = [text], model=model).data[0].embedding
    return np.array(response).astype(np.float32)


# faiss_index 불러오기
if os.path.exists(index_path):  # 파일 존재 O
    faiss_index = faiss.read_index(index_path)
    if faiss_index.d != 1536:
        raise RuntimeError("faiss_index의 차원이 1536이 아닙니다.")
else:  # 파일 존재 X - 인덱스 생성
    faiss_index = faiss.IndexFlatL2(1536)

# metadata 불러오기
if os.path.exists(metadata_path):
    metadata_path.read_text(encoding="utf-8")
else: metadata = []



# -------- 가게 임베딩 --------

class StoreAddRequest(BaseModel):
    name : str
    description : str
    market : str
    main_category : str
    sub_category : list[str]
    address : str

def doc_text(store: Dict[str, Any]):
    sub_category = ", ".join(store.get("sub_category", []))
    return f""" name : {store["name"]} /
                market : {store["market"]} /
                main_category : {store["main_category"]} /
                sub_category : {sub_category} /
                description : {store["description"]} /
                address : {store["address"]}
            """

@app.post("/add-store")
def add_store(request : StoreAddRequest):
    text = doc_text(request.model_dump())
    vector = get_embedding(text).reshape(1,-1)
    faiss_index.add(vector)
    metadata.append(request.model_dump())
    faiss.write_index(faiss_index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return { "count" : len(metadata) }



# -------- 코스 추천 --------

class CourseRecommendationRequest(BaseModel):
    market_name : str
    with_option : str
    atmosphere_option : str


@app.post("/course-recommendation")
def course_recommend(request:CourseRecommendationRequest):

    # 쿼리 정의 및 임베딩
    query = f"""{request.market_name} 안에 있는 가게들 중 분위기가 {request.atmosphere_option} 하고, 
                    {request.with_option}과 함께 가기 좋은 가게 위주로 6개 골라줘"""
    
    query_embedding = get_embedding(query).reshape(1,-1)

    # 오류 반환
    ntotal = getattr(faiss_index, "ntotal", 0)
    if ntotal != len(metadata):
        return {
            "coures_title" : "",
            "course_long_description" : "",
            "course_short_description" : "",
            "course_store":[],
            "error": "metadata 개수와 ntotal이 일치하지 않습니다."
        }
    if ntotal < 3:
        return {
            "coures_title" : "",
            "course_long_description" : "",
            "course_short_description" : "",
            "course_store":[],
            "error": "ntotal이 3 이하입니다."
        }
    
    store_list = ""
    #상위 6개 가게 추출
    k = min(6, ntotal)
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_stores = [metadata[i] for i in indices[0]]
    for i, store in enumerate(retrieved_stores):
        store_list += f"#{i+1} : {store['name']} - {store['description']} \n"

    # 응답 생성 요청
    data_schema = {
        "type": "object",
        "properties": {
            "course_title": {
                "type" : "string",
                "description" : "코스 제목"
            },
            "course_long_description": {
                "type" : "string",
                "description" : "코스 여러 줄 설명"
            },
            "course_short_description": {
                "type" : "string",
                "description" : "코스 한 줄 설명"
            },
            "course_store": {
                "type" : "array",
                "description" : "코스에 들어갈 가게 3가지",
                "items" : {
                    "type" : "object",
                    "properties" : {
                        "name" : {
                            "type" : "string",
                            "description" : "가게 이름"
                        }, 
                        "order" : {
                            "type" : "integer",
                            "description" : "코스 내 순서"
                        }
                    },
                    "required" : ["name", "order"]
                }
            }
        },
        "required": ["course_title", "course_long_description", "course_short_description", "course_store"]
    }
    
    user_prompt = f""" 가게 리스트: {store_list}
                        질문 : "가게 리스트 중 가게 3가지를 골라 코스를 만들어줘.
                        1. 분위기가 {request.atmosphere_option} 하고, 
                        {request.with_option}과 함께 가기 좋은 가게 위주로 골라줘.
                        2. 최적의 경로로 코스 내 방문 '순서'를 정해줘.
                        3. 카테고리가 되도록이면 겹치지 않게 코스를 만들어줘.
                        4. '코스의 제목', '코스 한 줄 설명', '코스 여러 줄 설명'을 작성해줘. 설명을 덧붙여서 자연스럽게 작성해줘.
                        5. 되도록이면 매번 다양한 구성으로 코스를 만들어줘"
                    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role" : "user",
                "content" : user_prompt
            }
        ],
        functions = [
            {
                "name" : "get_course_recommendation",
                "description": "코스 추천 불러오기",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "get_course_recommendation" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)
    return result
