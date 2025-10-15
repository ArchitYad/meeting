import streamlit as st
import tempfile
import subprocess
import os
from groq import Groq
import google.generativeai as genai

# Set API keys via Streamlit secrets or env variables
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

st.title("🎙️ Meeting Summarizer")

uploaded_file = st.file_uploader("Upload your meeting audio file", type=["mp3","wav","m4a","ogg","flac"])

def convert_to_wav(input_path):
    """Convert any audio file to WAV using ffmpeg"""
    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    ffmpeg_exe = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()
    if not ffmpeg_exe:
        st.error("FFmpeg not found. Please install ffmpeg in your environment.")
        return None
    subprocess.run([ffmpeg_exe, "-y", "-i", input_path, wav_temp.name], check=True)
    return wav_temp.name

def split_audio_wav(wav_path, chunk_ms=2*60*1000):
    """Split WAV into chunks (ms) using ffmpeg"""
    import math
    # Get duration in seconds
    result = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    chunks = []
    total_chunks = math.ceil(duration*1000 / chunk_ms)
    for i in range(total_chunks):
        start_ms = i * chunk_ms
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", str(start_ms/1000),
            "-t", str(chunk_ms/1000),
            out_file.name
        ], check=True)
        chunks.append(out_file.name)
    return chunks

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    wav_path = convert_to_wav(tmp_path)
    if wav_path:
        st.info("Processing audio... This may take a few minutes depending on length.")

        # Split audio into chunks
        chunk_files = split_audio_wav(wav_path)
        transcripts = []

        # Transcribe each chunk
        for idx, chunk_file in enumerate(chunk_files):
            with open(chunk_file, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(f.name, f.read()),
                    model="whisper-large-v3"
                )
            transcripts.append(transcription.text)

        full_transcript = "\n".join(transcripts)
        st.subheader("📝 Transcript")
        st.text_area("Full Transcript", full_transcript, height=300)

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
        response = gemini_model.generate_content(prompt)
        summary_text = response.text

        st.subheader("📄 Summary")
        st.text_area("Meeting Summary", summary_text, height=300)

        # Clean up temp files
        os.remove(tmp_path)
        os.remove(wav_path)
        for f in chunk_files:
            os.remove(f)
