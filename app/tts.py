from gtts import gTTS
import os
import uuid

AUDIO_OUTPUT_DIR = "tts_audios"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

def generate_tts(sentence: str) -> str:
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    tts = gTTS(text=sentence, lang='ko')
    tts.save(filepath)
    return filepath
