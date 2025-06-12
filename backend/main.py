from fastapi import FastAPI

from utils.notes_and_search import router as notes_search_router
from utils.log import router as logs_router
from utils.progress import router as progress_router

app = FastAPI()

app.include_router(notes_search_router)
app.include_router(logs_router)
app.include_router(progress_router)
