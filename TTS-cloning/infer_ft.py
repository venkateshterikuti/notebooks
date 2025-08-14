# infer_ft.py
import os, glob, torch, soundfile as sf
from TTS.tts.models.xtts import Xtts
from TTS.utils.synthesizer import Synthesizer

CKPT_DIR = "runs/xtts_ft/checkpoints"
REF = "ref.wav"
TEXT = "This is a quick test from my finetuned checkpoint."
SR = 22050

# pick the newest checkpoint
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
assert ckpts, f"No checkpoints found in {CKPT_DIR}"
ckpt = ckpts[-1]
print("Loading:", ckpt)

# restore model
model = Xtts.init_from_config(None)     # config is embedded in checkpoint
model.load_checkpoint(ckpt, eval=True)

if torch.cuda.is_available():
    model.cuda()

syn = Synthesizer(model, None, None)    # vocoder is integrated for XTTS

wav = syn.tts(text=TEXT, speaker_wav=REF, language="en")
sf.write("ft_output.wav", wav, SR)
print("Saved ft_output.wav")
