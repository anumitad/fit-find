import streamlit as st
import httpx
from datetime import date
import base64


API_URL = "http://backend:8000"

st.title("FitFind — Add Text to Vector DB")

tab1, tab2, tab3, tab4 = st.tabs(["Notes", "Search", "Training & Nutrition Logs", "Progress & Prediction Tracking"])


with tab1: 
    st.subheader("Enter personal notes")
    user_text = st.text_area("Enter fitness-related notes, transcript, or content:")

    if st.button("Submit"):
        if user_text.strip():
            with st.spinner("Sending to backend..."):
                try:
                    response = httpx.post(f"{API_URL}/add_text", json={"text": user_text})
                    if response.status_code == 200:
                        st.success("Text stored successfully in Qdrant!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")
        else:
            st.warning("Text input cannot be empty.")


    st.header("Upload Audio or Video")
    uploaded_file = st.file_uploader("Upload an audio/video file", type=["mp3", "wav", "mp4", "m4a", "webm"])

    if uploaded_file is not None:
        if st.button("Transcribe & Store"):
            with st.spinner("Uploading and transcribing..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = httpx.post(f"{API_URL}/upload_media", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Transcription successful and stored!")
                        st.markdown(f"**Transcribed Text:**\n\n{result['text']}")
                    else:
                        st.error(f"Backend error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to send file: {e}")


    st.subheader("Ingest a YouTube Video")
    video_url = st.text_input("Paste a YouTube video URL")

    if st.button("Ingest Video"):
        with st.spinner("ingesting video..."):
            if video_url.strip():
                response = httpx.post(f"{API_URL}/ingest/youtube", json={"url": video_url}, timeout=360.0)
                if response.status_code == 200:
                    st.success("Video ingested!")
                else:
                    st.error("Error: " + response.text)


with tab2: 
    def search_notes(query):
        response = httpx.post(f"{API_URL}/search", json={"query": query})
        if response.status_code == 200:
            return response.json()
        return []

    def search_gemini(query):
        response = httpx.post(f"{API_URL}/search_gemini", json={"query": query})
        if response.status_code == 200:
            return response.json().get("answer")
        return "Error calling Gemini API"

    # Section: Search
    st.header("Search")

    query = st.text_input("Enter your fitness question:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Search Notes"):
            if not query.strip():
                st.warning("Please enter a search query")
            else:
                with st.spinner("Searching..."):
                    results = search_notes(query)
                    st.subheader("Notes Search Results:")
                    if results:
                        for idx, res in enumerate(results):
                            st.write(f"{idx+1}. {res['text']} (score: {res['score']:.2f})")
                    else:
                        st.write("No results found.")

    with col2:
        if st.button("Search Web (Gemini)"):
            with st.spinner("Searching..."):
                if not query.strip():
                    st.warning("Please enter a search query")
                else:
                    answer = search_gemini(query)
                    st.subheader("Gemini Web Search Answer:")
                    st.write(answer)


with tab3:
    with st.expander("Log Body Metrics"):
        st.header("Log Body Metrics")

        bm_date = st.date_input("Date", value=date.today())
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
        body_fat = st.number_input("Body Fat (%)", min_value=0.0, max_value=100.0, step=0.1)
        muscle_mass = st.number_input("Muscle Mass (%)", min_value=0.0, step=0.1)

        if st.button("Save Body Metrics", key="body_metrics"):
            data = {
                "date": str(bm_date),
                "weight": weight,
                "body_fat": body_fat,
                "muscle_mass": muscle_mass
            }
            res = httpx.post(f"{API_URL}/body-metrics", json=data)
            if res.status_code == 200:
                st.success("Body metrics saved!")
            else:
                st.error(f"Error: {res.text}")

    with st.expander("Log Workout"):
        st.header("Log Workout")

        wo_date = st.date_input("Date", key="wo_date", value=date.today())
        exercise = st.text_input("Exercise")

        num_sets = st.number_input("Number of Sets", min_value=1, max_value=10, step=1, key="num_sets")

        sets = []
        for i in range(num_sets):
            st.markdown(f"**Set {i+1}**")
            reps = st.number_input(f"Reps for set {i+1}", min_value=1, step=1, key=f"reps_{i}")
            weight = st.number_input(f"Weight (kg) for set {i+1}", min_value=0.0, step=0.1, key=f"weight_{i}")
            sets.append({"reps": reps, "weight": weight})

        if st.button("Save Workout", key="save_workout"):
            workout_data = {
                "date": str(wo_date),
                "exercise": exercise,
                "sets": sets
            }
            res = httpx.post(f"{API_URL}/workouts", json=workout_data)
            if res.status_code == 200:
                st.success("Workout saved!")
            else:
                st.error(f"Error: {res.text}")

    with st.expander("Log Nutrition"):
        st.header("Log Nutrition")

        nutri_date = st.date_input("Date", key="nutri_date", value=date.today())
        food = st.text_input("Food")
        calories = st.number_input("Calories", min_value=0, step=1)

        if st.button("Save Nutrition Log", key="save_nutrition"):
            nutrition_data = {
                "date": str(nutri_date),
                "food": food,
                "calories": calories
            }
            res = httpx.post(f"{API_URL}/nutrition", json=nutrition_data)
            if res.status_code == 200:
                st.success("Nutrition log saved!")
            else:
                st.error(f"Error: {res.text}")


with tab4:
    st.title("Progress & Prediction Tracker")

    metric = st.selectbox("Choose metric", ["weight", "body_fat", "muscle_mass"])
    months = st.slider("How many months to look back and predict forward?", 1, 6, 2)

    if st.button("Generate Progress Report"):
        with st.spinner("Generating report..."):
            response = httpx.post(f"{API_URL}/progress",params={"metric_type": metric, "months": months})

            if response.status_code == 200:
                data = response.json()
                img_bytes = base64.b64decode(data["image_base64"])
                st.image(img_bytes, caption=f"{metric.capitalize()} Progress (±{months} months)")
            else:
                st.error(f"Backend error: {response.text}")            
