import os
import sqlite3
import oracledb
import io
import base64


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv


router = APIRouter(prefix="/progress", tags=["progress"])

load_dotenv()

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


@router.post("/progress")
async def get_progress_image(metric_type: str,months: int):
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
   