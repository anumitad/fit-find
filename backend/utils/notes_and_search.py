# notes.py
import uuid
import tempfile
import whisper
import yt_dlp
import os
import re

from fastapi import File, UploadFile, APIRouter, HTTPException
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from google import genai

load_dotenv()


router = APIRouter(prefix="/notes_and_search", tags=["notes_and_search"])

model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(host="qdrant", port=6333)

COLLECTION_NAME = "fitfind-text"

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

whisper_model = whisper.load_model("base")

client = genai.Client(api_key=os.getenv("GEMINI_TOKEN"))


class TextInput(BaseModel):
    text: str

class YouTubeIngestRequest(BaseModel):
    url: str

class SearchQuery(BaseModel):
    query: str


def split_into_chunks(text: str, max_length: int = 500) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@router.post("/add_text")
async def add_text(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Empty text input")
    

    chunks = split_into_chunks(input.text)

    points = []
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "source": "manual notes",
                "chunk_index": idx,
                # optionally:
                # "timestamp": calculate_estimated_timestamp(idx, total_chunks)
            }
        )
        points.append(point)

    qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
    )


    return {"message": "Text stored in Qdrant"}

@router.post("/upload_media")
async def upload_media(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=True, suffix=file.filename) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        # Transcribe using Whisper
        result = whisper_model.transcribe(tmp.name)
        transcription = result.get("text", "").strip()

        if not transcription:
            raise HTTPException(status_code=500, detail="Transcription failed")
        

        chunks = split_into_chunks(transcription)

        points = []
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file.filename,
                    "chunk_index": idx,
                    # optionally:
                    # "timestamp": calculate_estimated_timestamp(idx, total_chunks)
                }
            )
            points.append(point)

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        return {"message": "Stored successfully", "text": transcription}
 
@router.post("/ingest/youtube")  
async def ingest_youtube_video(data: YouTubeIngestRequest):
    url = data.url

    # Setup yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '/tmp/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file_path = ydl.prepare_filename(info_dict).replace(info_dict['ext'], 'mp3')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"yt_dlp error: {str(e)}")

    # Transcribe using Whisper
    try:
        result = whisper_model.transcribe(audio_file_path)
        transcription = result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcription error: {str(e)}")

    if not transcription:
        raise HTTPException(status_code=500, detail="Transcription returned empty text")


    chunks = split_into_chunks(transcription)

    points = []
    # Create embeddings
    #embedding = model.encode(transcription).tolist()

    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        # Store in Qdrant
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "source": url,
                "title": info_dict.get("title", "")
            }
        )
        points.append(point)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {"message": "YouTube video ingested successfully", "title": info_dict.get("title", ""), "transcription": transcription}

@router.post("/search")
async def search(query: SearchQuery):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Empty search query")

    query_embedding = model.encode(query.query).tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=3
    )

    return [
    {
        "text": result.payload.get("text"),
        "score": result.score,
        "source": result.payload.get("source"),
        "type": result.payload.get("type"),
        "chunk_index": result.payload.get("chunk_index"),
        "timestamp": result.payload.get("timestamp")
    }
    for result in results
]

@router.post("/search_gemini")
async def search_gemini(query: SearchQuery) -> dict:
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Empty search query")

    prompt = f"Answer this fitness question concisely:\n\n{query.query}"
    answer = response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
    )

    return {"answer": answer}

