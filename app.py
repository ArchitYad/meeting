import os
import tempfile
import subprocess
import streamlit as st
from groq import Groq
import google.generativeai as genai

# ------------------- SECRETS -------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# ------------------- INIT CLIENTS -------------------
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------- STREAMLIT UI -------------------
st.title("🎙️ Meeting Audio Summarizer")
st.write("Upload your meeting audio (mp3, wav, m4a, etc.) and get a transcript & summary.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

def convert_to_wav(input_path, output_path):
    """Convert audio to WAV using ffmpeg subprocess"""
    subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], check=True)

if uploaded_file:
    st.info("Processing audio… this may take a few moments.")
    transcript_text = ""
    summary_text = ""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Convert to WAV
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        convert_to_wav(temp_path, wav_path)

        # Split audio into 2-min chunks using ffmpeg
        chunk_dir = tempfile.mkdtemp()
        subprocess.run([
            "ffmpeg", "-i", wav_path, "-f", "segment", "-segment_time", "120",
            "-c", "copy", os.path.join(chunk_dir, "chunk%03d.wav")
        ], check=True)

        # Transcribe each chunk
        transcripts = []
        for chunk_file in sorted(os.listdir(chunk_dir)):
            chunk_path = os.path.join(chunk_dir, chunk_file)
            with open(chunk_path, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(chunk_file, f.read()),
                    model="whisper-large-v3"
                )
            transcripts.append(transcription.text)

        transcript_text = "\n".join(transcripts)
        st.subheader("📄 Transcript")
        st.text_area("Transcript", transcript_text, height=300)

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
        response = gemini_model.generate_content(prompt)
        summary_text = response.text
        st.subheader("📝 Summary")
        st.text_area("Summary", summary_text, height=300)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
