# batch_infer.py
from TTS.api import TTS
from pathlib import Path

MODEL = "coqui/XTTS-v2"
REF = "ref.wav"   # your reference voice clip
LANG = "en"       # English

# read input lines
lines = [ln.strip() for ln in Path("lines.txt").read_text().splitlines() if ln.strip()]

tts = TTS(MODEL)  # will auto-download the model the first time
for i, line in enumerate(lines, 1):
    out = f"out_{i:02d}.wav"
    tts.tts_to_file(text=line, speaker_wav=REF, language=LANG, file_path=out)
    print(f"[OK] {out}: {line}")
print("Done.")
