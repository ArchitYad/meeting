# meeting_summarizer_streamlit.py

import os
import tempfile
import streamlit as st
from groq import Groq
import google.generativeai as genai
from pydub import AudioSegment
import imageio_ffmpeg as ffmpeg

# --- Set ffmpeg for pydub ---
AudioSegment.converter = ffmpeg.get_ffmpeg_exe()

# --- Load API keys from environment variables ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- Initialize clients ---
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- Streamlit UI ---
st.set_page_config(page_title="Meeting Summarizer", layout="centered")
st.title("📝 Meeting Summarizer")
st.write("Upload a meeting audio file and get transcript + summary.")

uploaded_file = st.file_uploader("Upload Audio (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.info("Processing audio... This may take a while for long meetings.")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Convert to WAV if needed
        if not temp_path.endswith(".wav"):
            sound = AudioSegment.from_file(temp_path)
            wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sound.export(wav_temp.name, format="wav")
            temp_path = wav_temp.name

        # Split audio into 2-minute chunks
        audio = AudioSegment.from_wav(temp_path)
        chunk_length_ms = 2 * 60 * 1000  # 2 minutes
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        transcripts = []
        for idx, chunk in enumerate(chunks):
            chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            chunk.export(chunk_file.name, format="wav")
            with open(chunk_file.name, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(f.name, f.read()),
                    model="whisper-large-v3"
                )
            transcripts.append(transcription.text)

        # Aggregate full transcript
        transcript_text = "\n".join(transcripts)

        # Summarize transcript
        prompt = f"""
        You are a professional meeting assistant.
        Summarize this transcript into:
        - Key Decisions
        - Action Items
        - Discussion Points

        Transcript:
        {transcript_text}
        """
        summary_response = gemini_model.generate_content(prompt)
        summary_text = summary_response.text

        # Display results
        st.subheader("🗒 Transcript")
        st.text_area("Transcript", value=transcript_text, height=300)
        st.subheader("💡 Summary & Action Items")
        st.text_area("Summary", value=summary_text, height=300)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
