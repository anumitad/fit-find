from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid
from fastapi import File, UploadFile
import tempfile
import whisper
import re
from typing import List
from dotenv import load_dotenv
from google import genai
import os
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from typing import Dict, Any
from typing import List, Optional
from datetime import date
import sqlite3
import oracledb
import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64



load_dotenv()

app = FastAPI()


# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant (using Docker service name)
qdrant = QdrantClient(host="qdrant", port=6333)

client = genai.Client(api_key=os.getenv("GEMINI_TOKEN"))

USE_ORACLE = os.getenv("USE_ORACLE", "false").lower() == "true"

if USE_ORACLE:
    #oracledb.init_oracle_client(config_dir=os.getenv("ORACLE_WALLET_PATH"))
    #oracledb.init_oracle_client()


    conn = oracledb.connect(
    user="admin",
    password=os.getenv("ORACLE_DB_PASSWORD"),
    dsn=os.getenv("ORACLE_DB_DSN"), 
    config_dir=os.getenv("ORACLE_WALLET_PATH")
    )

else:
    conn = sqlite3.connect("fitness.db", check_same_thread=False)

cursor = conn.cursor()

COLLECTION_NAME = "fitfind-text"

# Ensure collection exists (create if not)
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)


class TextInput(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str

class YouTubeIngestRequest(BaseModel):
    url: str

class BodyMetrics(BaseModel):
    date: date
    weight: Optional[float]
    body_fat: Optional[float]
    muscle_mass: Optional[float]

class WorkoutSet(BaseModel):
    reps: int
    weight: Optional[float] = None

class WorkoutEntry(BaseModel):
    date: date
    exercise: str
    sets: List[WorkoutSet]

class NutritionLog(BaseModel):
    date: date
    food: str
    calories: Optional[int]


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

@app.post("/add_text")
def add_text(input: TextInput):
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


@app.post("/search")
def search(query: SearchQuery):
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

# Load whisper model once (base is a good default)
whisper_model = whisper.load_model("base")

@app.post("/upload_media")
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
    

def call_gemini_api(prompt: str) -> str:
    # Example: call your Gemini API here and return the generated text
    # This is a placeholder
    response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
    )


    return f"Gemini answer for prompt: {response}"


@app.post("/search_gemini")
def search_gemini(query: SearchQuery) -> dict:
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Empty search query")

    prompt = f"Answer this fitness question concisely:\n\n{query.query}"
    answer = call_gemini_api(prompt)

    return {"answer": answer}


@app.post("/ingest/youtube")  
def ingest_youtube_video(data: YouTubeIngestRequest):
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

if not USE_ORACLE:
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS body_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_column TEXT NOT NULL,
        weight REAL,
        body_fat REAL,
        muscle_mass REAL
    );

    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_column TEXT NOT NULL,
        exercise TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS workout_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_id INTEGER,
        set_number INTEGER,
        reps INTEGER,
        weight REAL,
        FOREIGN KEY (workout_id) REFERENCES workouts(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS nutrition_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_column TEXT NOT NULL,
        food TEXT NOT NULL,
        calories INTEGER
    );
    """)
else:
    tables = [
        """
        BEGIN
        EXECUTE IMMEDIATE '
            CREATE TABLE body_metrics (
                id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY PRIMARY KEY,
                date_column DATE NOT NULL,
                weight NUMBER,
                body_fat NUMBER,
                muscle_mass NUMBER
            )
        ';
        EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN -- ORA-00955: name is already used by an existing object
                RAISE;
            END IF;
        END;
        """,

        """
        BEGIN
        EXECUTE IMMEDIATE '
            CREATE TABLE workouts (
                id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY PRIMARY KEY,
                date_column DATE NOT NULL,
                exercise VARCHAR2(255) NOT NULL
            )
        ';
        EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN
                RAISE;
            END IF;
        END;
        """,

        """
        BEGIN
        EXECUTE IMMEDIATE '
            CREATE TABLE workout_sets (
                id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY PRIMARY KEY,
                workout_id NUMBER,
                set_number NUMBER,
                reps NUMBER,
                weight NUMBER,
                CONSTRAINT fk_workout
                    FOREIGN KEY (workout_id) REFERENCES workouts(id) ON DELETE CASCADE
            )
        ';
        EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN
                RAISE;
            END IF;
        END;
        """,

        """
        BEGIN
        EXECUTE IMMEDIATE '
            CREATE TABLE nutrition_logs (
                id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY PRIMARY KEY,
                date_column DATE NOT NULL,
                food VARCHAR2(255) NOT NULL,
                calories NUMBER
            )
        ';
        EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN
                RAISE;
            END IF;
        END;
        """
    ]

    for table_sql in tables:
        cursor.execute(table_sql)

    sequence_statements = [
        "CREATE SEQUENCE body_metrics_seq START WITH 1 INCREMENT BY 1 NOCACHE",
        "CREATE SEQUENCE workouts_seq START WITH 1 INCREMENT BY 1 NOCACHE",
        "CREATE SEQUENCE workout_sets_seq START WITH 1 INCREMENT BY 1 NOCACHE",
        "CREATE SEQUENCE nutrition_logs_seq START WITH 1 INCREMENT BY 1 NOCACHE"
    ]

    for stmt in sequence_statements:
        try:
            cursor.execute(stmt)
            print(f"Executed: {stmt}")
        except oracledb.Error as e:
            # Optional: ignore error if sequence already exists
            error_obj, = e.args
            if "ORA-00955" in error_obj.message:
                print(f"Sequence already exists: {stmt}")
            else:
                raise

    
conn.commit()

@app.post("/body-metrics")
def add_body_metrics(metrics: BodyMetrics):
    if not USE_ORACLE:
        cursor.execute(
            "INSERT INTO body_metrics (date_column, weight, body_fat, muscle_mass) VALUES (?, ?, ?, ?)",
            (metrics.date.isoformat(), metrics.weight, metrics.body_fat, metrics.muscle_mass)
        )
    
    else: 
        cursor.execute(
            """
            INSERT INTO body_metrics (id, date_column, weight, body_fat, muscle_mass)
            VALUES (body_metrics_seq.NEXTVAL, :date_column, :weight, :body_fat, :muscle_mass)
            """,
            {
                "date_column": metrics.date,
                "weight": metrics.weight,
                "body_fat": metrics.body_fat,
                "muscle_mass": metrics.muscle_mass
            }
        )

    conn.commit()
    return {"status": "success", "message": "Body metrics saved."}


@app.post("/workouts")
def add_workout(entry: WorkoutEntry):
    if not USE_ORACLE:
        cursor.execute(
            "INSERT INTO workouts (date_column, exercise) VALUES (?, ?)",
            (entry.date.isoformat(), entry.exercise)
        )
        workout_id = cursor.lastrowid

        for i, s in enumerate(entry.sets, start=1):
                cursor.execute(
                    "INSERT INTO workout_sets (workout_id, set_number, reps, weight) VALUES (?, ?, ?, ?)",
                    (workout_id, i, s.reps, s.weight)
                )  
    else: 
        cursor.execute(
            "INSERT INTO workouts (id, date_column, exercise) VALUES (workouts_seq.NEXTVAL, :date_column, :exercise)",
            {"date_column": entry.date, "exercise": entry.exercise}
        )
        # Get the generated ID of the inserted workout
        cursor.execute("SELECT workouts_seq.CURRVAL FROM dual")
        workout_id = cursor.fetchone()[0]

        for i, s in enumerate(entry.sets, start=1):
            cursor.execute(
                """
                INSERT INTO workout_sets (id, workout_id, set_number, reps, weight)
                VALUES (workout_sets_seq.NEXTVAL, :workout_id, :set_number, :reps, :weight)
                """,
                {
                    "workout_id": workout_id,
                    "set_number": i,
                    "reps": s.reps,
                    "weight": s.weight
                }
            )   

    conn.commit()
    return {"status": "success", "message": "Workout saved."}


@app.post("/nutrition")
def add_nutrition_log(log: NutritionLog):
    if not USE_ORACLE:
        cursor.execute(
            "INSERT INTO nutrition_logs (date_column, food, calories) VALUES (?, ?, ?)",
            (log.date.isoformat(), log.food, log.calories)
        )

    else:
        cursor.execute(
            "INSERT INTO nutrition_logs (id, date_column, food, calories) VALUES (nutrition_logs_seq.NEXTVAL, :date_column, :food, :calories)",
            {"date_column": log.date, "food": log.food, "calories": log.calories}
        )

    conn.commit()
    return {"status": "success", "message": "Nutrition log saved."}

@app.post("/progress")
def get_progress_image(metric_type: str,months: int):
    if not USE_ORACLE:
        query = f"""
            SELECT 
                strftime('%Y-%m-%d', date_column) AS ds,
                {metric_type} AS y
            FROM 
                body_metrics
            WHERE 
                date_column >= date('now', ?)
                AND {metric_type} IS NOT NULL
            ORDER BY 
                date_column;
        """

        df = pd.read_sql(query, conn, params=(f'-{months} months'))
 

    else:
        query = f"""
            SELECT 
                TO_CHAR(date_column, 'YYYY-MM-DD') AS ds,
                {metric_type} AS y
            FROM 
                body_metrics
            WHERE 
                date_column >= ADD_MONTHS(SYSDATE, -:1)
                AND {metric_type} IS NOT NULL
            ORDER BY 
                date_column
        """

        df = pd.read_sql(query, conn, params=[months])

    #print(df.head())
    
    df.columns = [col.lower() for col in df.columns]

    model = Prophet()
    model.fit(df)

    # Step 3: Forecast future `months`
    future_days = months * 30
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

    # Step 4: Plot results
    fig = model.plot(forecast)
    plt.title(f"{metric_type.capitalize()} Prediction")

    # Step 5: Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return JSONResponse(content={"image_base64": encoded})
   