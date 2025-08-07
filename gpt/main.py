from fastapi import FastAPI
from gpt.translate import app as translate_app
from gpt.store_description import app as description_app

app = FastAPI()

app.mount("/gpt/translate", translate_app)
app.mount("/gpt/store-description", description_app)