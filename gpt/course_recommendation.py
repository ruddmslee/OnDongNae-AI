from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import os, json
from dotenv import load_dotenv
import numpy as np
import faiss
from typing import Dict, Any, List, Optional

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

def get_normalized_embedding(text):
    vec = get_embedding(text)
    return vec / (np.linalg.norm(vec) + 1e-12)

# faiss_index 불러오기
if os.path.exists(index_path):  # 파일 존재 O
    faiss_index = faiss.read_index(index_path)
    if faiss_index.d != 1536:
        raise RuntimeError("faiss_index의 차원이 1536이 아닙니다.")
else:  # 파일 존재 X - 인덱스 생성
    faiss_index = faiss.IndexFlatIP(1536)

# metadata 불러오기
metadata: List[Dict[str, Any]] = []
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else: metadata = []

# -------- 응답 처리 --------
class CourseStore(BaseModel):
    id: int
    name: str
    order: int

class CourseRecommendationResponse(BaseModel):
    course_title: str
    course_long_description: str
    course_short_description: str
    course_store: List[CourseStore]

class ApiResponse(BaseModel):
    success : bool
    data: Optional[CourseRecommendationResponse] = None
    error: Optional[str] = None

def ok(data: CourseRecommendationResponse):
    return ApiResponse(success=True, data=data)
def error(msg:str):
    return ApiResponse(success=False, error=msg)


# -------- 가게 임베딩 --------

class StoreAddRequest(BaseModel):
    id: int
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
    vector = get_normalized_embedding(text).reshape(1,-1)
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
    query = f"""Select 6 stores located in {request.market_name} that have a {request.atmosphere_option} atmosphere
                    and are good to visit with {request.with_option}"""
    
    query_embedding = get_normalized_embedding(query).reshape(1,-1)

    # 오류 반환
    ntotal = getattr(faiss_index, "ntotal", 0)
    if ntotal != len(metadata):
        return error("metadata 개수와 ntotal이 일치하지 않습니다.")
    if ntotal < 3:
        return error("ntotal이 3 이하입니다.")
    
    store_list = []
    #상위 6개 가게 추출
    k = min(6, ntotal)
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_stores = [metadata[i] for i in indices[0]]
    for i, store in enumerate(retrieved_stores):
        store_list.append({"id": store["id"],
                            "name": store["name"],
                            "description" : store["description"]})
        
    store_list_json = json.dumps(store_list, ensure_ascii=False)

    # 응답 생성 요청
    data_schema = {
        "type": "object",
        "properties": {
            "course_title": {
                "type" : "string",
                "description" : "course title"
            },
            "course_long_description": {
                "type" : "string",
                "description" : "course description about three lines long"
            },
            "course_short_description": {
                "type" : "string",
                "description" : "course description in one line"
            },
            "course_store": {
                "type" : "array",
                "description" : "3 stores to include in the course",
                "items" : {
                    "type" : "object",
                    "properties" : {
                        "id" : {
                            "type" : "integer",
                            "description" : "store id"
                        },
                        "name" : {
                            "type" : "string",
                            "description" : "store name"
                        }, 
                        "order" : {
                            "type" : "integer",
                            "description" : "store order in the course"
                        }
                    },
                    "required" : ["id", "name", "order"]
                }
            }
        },
        "required": ["course_title", "course_long_description", "course_short_description", "course_store"]
    }
    
    user_prompt = f""" Store list: {store_list_json}
                        Question : "From the given store list, select 3 stores to create a course. (Make sure to use only the candidate store IDs)
                        1. Choose stores that have a "{request.atmosphere_option}" atmosphere and
                        good to visit with {request.with_option}.
                        2. Determine the visiting 'order' within the course to create the optional route.
                        3. Try to select stores with different categories if possible.
                        4. Provide 'course_title', 'course_short_description' (one line), and 'course_long_description' (about three lines) in Korean.
                        Write the descriptions naturally with context.
                        5. Try to make the composition of the course different each time you generate it.
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
                "description": "Retrieve a course recommendation",
                "parameters": data_schema
            }
        ],
        function_call={ "name" : "get_course_recommendation" }
    )

    result = json.loads(response.choices[0].message.function_call.arguments)

    try :
        data = CourseRecommendationResponse(** result)
    except ValidationError as e:
        return error("응답 형식 검증에 실패했습니다.")
    
    return ok(data)
