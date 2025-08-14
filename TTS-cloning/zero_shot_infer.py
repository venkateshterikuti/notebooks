from TTS.api import TTS

# Auto-downloads XTTS-v2 on first use
tts = TTS(model_name="coqui/XTTS-v2")  # HF model id

LANG = "en"
TEXT = "Hello! This is a quick zero-shot demo using your voice sample."
REF  = "ref.wav"  # 6–30s clean reference

tts.tts_to_file(
    text=TEXT,
    speaker_wav=REF,
    language=LANG,
    file_path="output.wav"
)
print("Saved output.wav")
