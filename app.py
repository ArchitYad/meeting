import os
import tempfile
import streamlit as st
from groq import Groq
import google.generativeai as genai
from pydub.audio_segment import AudioSegment
import imageio_ffmpeg as ffmpeg

# Force pydub to use ffmpeg from imageio-ffmpeg
AudioSegment.converter = ffmpeg.get_ffmpeg_exe()

# Load API keys from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

st.title("📋 Meeting Summarizer")

uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","m4a"])
if uploaded_file:
    with st.spinner("Processing..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Convert to WAV
        if not temp_path.endswith(".wav"):
            audio = AudioSegment.from_file(temp_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
                audio.export(wav_tmp.name, format="wav")
                temp_path = wav_tmp.name

        # Split audio into 2-minute chunks
        audio = AudioSegment.from_wav(temp_path)
        chunk_length_ms = 2*60*1000
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        transcripts = []
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_tmp:
                chunk.export(chunk_tmp.name, format="wav")
                with open(chunk_tmp.name, "rb") as f:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(chunk_tmp.name, f.read()),
                        model="whisper-large-v3"
                    )
            transcripts.append(transcription.text)

        full_transcript = "\n".join(transcripts)
        st.subheader("📝 Transcript")
        st.text_area("Transcript", full_transcript, height=300)

        # Summarize with Gemini
        prompt = f"""
You are a professional meeting assistant.
Summarize this transcript into:
- Key Decisions
- Action Items
- Discussion Points

Transcript:
{full_transcript}
"""
        summary = gemini_model.generate_content(prompt).text
        st.subheader("📌 Summary")
        st.text_area("Summary", summary, height=300)
