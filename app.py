import os
import tempfile
import streamlit as st
from groq import Groq
import google.generativeai as genai

# Use a forked version of pydub utils that doesn't import pyaudioop
from pydub.audio_segment import AudioSegment
import imageio_ffmpeg as ffmpeg

# Force pydub to use ffmpeg from imageio-ffmpeg
AudioSegment.converter = ffmpeg.get_ffmpeg_exe()

# --- Load API keys from Streamlit secrets ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- Initialize clients ---
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- Streamlit UI ---
st.title("📋 Meeting Summarizer")
st.write("Upload your meeting audio (mp3, wav, m4a) and get transcript + summary + action items.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","m4a"])

if uploaded_file:
    with st.spinner("Processing audio..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Convert to WAV if needed
        if not temp_path.endswith(".wav"):
            audio = AudioSegment.from_file(temp_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_temp:
                audio.export(wav_temp.name, format="wav")
                temp_path = wav_temp.name

        # Split audio into 2-min chunks
        audio = AudioSegment.from_wav(temp_path)
        chunk_length_ms = 2 * 60 * 1000  # 2 minutes
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        # Transcribe each chunk
        transcripts = []
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                with open(chunk_file.name, "rb") as f:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(f.name, f.read()),
                        model="whisper-large-v3"
                    )
            transcripts.append(transcription.text)

        # Aggregate transcript
        transcript_text = "\n".join(transcripts)
        st.subheader("📝 Transcript")
        st.text_area("Transcript", transcript_text, height=300)

        # Summarize with Gemini
        prompt = f"""
You are a professional meeting assistant.
Summarize this transcript into:
- Key Decisions
- Action Items
- Discussion Points

Transcript:
{transcript_text}
"""
        response = gemini_model.generate_content(prompt)
        summary_text = response.text

        st.subheader("📌 Summary / Action Items")
        st.text_area("Summary", summary_text, height=300)
