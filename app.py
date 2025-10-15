import os
import io
import streamlit as st
from groq import Groq
import google.generativeai as genai

# ---------------------------
# Load API keys
# ---------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Meeting Summarizer with Chunking")

uploaded_file = st.file_uploader("Upload your audio file (mp3, m4a, wav)", type=["mp3", "m4a", "wav"])

if uploaded_file is not None:
    st.info("Processing audio… this may take a few moments.")
    
    # Define chunk size in bytes (e.g., ~5MB per chunk)
    CHUNK_SIZE = 5 * 1024 * 1024
    audio_bytes = uploaded_file.read()
    total_size = len(audio_bytes)

    transcripts = []
    try:
        for start in range(0, total_size, CHUNK_SIZE):
            chunk_bytes = audio_bytes[start:start+CHUNK_SIZE]
            chunk_file = io.BytesIO(chunk_bytes)
            chunk_file.name = uploaded_file.name  # Groq needs a filename

            # Transcribe chunk
            transcription = groq_client.audio.transcriptions.create(
                file=(chunk_file.name, chunk_file.read()),
                model="whisper-large-v3"
            )
            transcripts.append(transcription.text)

        # Aggregate transcripts
        full_transcript = "\n".join(transcripts)
        st.subheader("Transcript")
        st.text_area("Transcript", full_transcript, height=200)

        # Summarize
        prompt = f"""
        You are a professional meeting assistant.
        Summarize this transcript into:
        - Key Decisions
        - Action Items
        - Discussion Points

        Transcript:
        {full_transcript}
        """
        response = gemini_model.generate_content(prompt)
        st.subheader("Summary")
        st.text_area("Summary", response.text, height=200)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
