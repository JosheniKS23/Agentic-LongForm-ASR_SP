import os
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer, wil, wip
import torch
import soundfile as sf

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==============================
# CONFIG
# ==============================
FOLDER = r"C:\Users\joshe\OneDrive - Amrita vishwa vidyapeetham\_collegefiles\SEM_files_\Sem_4\SP\project_endsem\amicorpus\IS1000a\audio"

BASE_CHUNK = 15
SMALL_CHUNK = 5
LARGE_CHUNK = 25

LOW_CONF = -1.0
HIGH_CONF = -0.6

# Add reference manually if available
REFERENCE_TEXTS = {
    # "IS1000a.Headset-0.wav": "your reference text here"
}

# ==============================
# LOAD MODELS
# ==============================
print("Loading models...")
tiny_model = whisper.load_model("tiny").to(device)
base_model = whisper.load_model("base").to(device)

# ==============================
# AGENT CONTROLLER
# ==============================
class AgentController:
    def __init__(self):
        self.prev_conf = None

    def decide_chunk_size(self):
        if self.prev_conf is None:
            return BASE_CHUNK
        if self.prev_conf < LOW_CONF:
            return SMALL_CHUNK
        elif self.prev_conf > HIGH_CONF:
            return LARGE_CHUNK
        return BASE_CHUNK

    def decide_model(self):
        if self.prev_conf is None:
            return "both"
        if self.prev_conf > HIGH_CONF:
            return "tiny"
        return "both"

    def need_redecode(self, conf):
        return conf < LOW_CONF

    def update(self, conf):
        self.prev_conf = conf

# ==============================
# SIMPLE SPEAKER SPLIT
# ==============================
def simple_speaker_split(text):
    sentences = text.split(".")
    result = []
    speaker = 0

    for s in sentences:
        if s.strip():
            result.append(f"SPEAKER_{speaker}: {s.strip()}")
            speaker = 1 - speaker

    return result

# ==============================
# SPLIT AUDIO
# ==============================
def split_audio(audio, sr, duration):
    size = int(duration * sr)
    return [audio[i:i+size] for i in range(0, len(audio), size)]

# ==============================
# ASR AGENT
# ==============================
def asr_agent(chunk, sr, mode):
    sf.write("temp.wav", chunk, sr)

    if mode == "tiny":
        return [tiny_model.transcribe("temp.wav", fp16=(device=="cuda"))]
    else:
        return [
            tiny_model.transcribe("temp.wav", fp16=False),
            base_model.transcribe("temp.wav", fp16=(device=="cuda"))
        ]

# ==============================
# CONFIDENCE
# ==============================
def get_conf(res):
    if res.get("segments"):
        return np.mean([s["avg_logprob"] for s in res["segments"]])
    return -2.0

# ==============================
# MODEL SELECTION
# ==============================
def selection_agent(results):
    best_text = ""
    best_conf = -2.0

    for r in results:
        c = get_conf(r)
        if c > best_conf:
            best_conf = c
            best_text = r["text"]

    return best_text, best_conf

# ==============================
# METRICS
# ==============================
def evaluate(file, pred_text):
    if file not in REFERENCE_TEXTS:
        print("No reference → skipping metrics")
        return

    ref = REFERENCE_TEXTS[file]

    print("\n📊 METRICS:")
    print("WER:", wer(ref, pred_text))
    print("WIL:", wil(ref, pred_text))
    print("WIP:", wip(ref, pred_text))

# ==============================
# PROCESS FILE
# ==============================
def process_file(file):
    print(f"\nProcessing: {file}")

    agent = AgentController()
    path = os.path.join(FOLDER, file)

    audio, sr = librosa.load(path, sr=16000, mono=True)

    pointer = 0
    final_text = ""
    confidences = []

    while pointer < len(audio):

        chunk_size = agent.decide_chunk_size()
        chunk_len = int(chunk_size * sr)

        chunk = audio[pointer:pointer+chunk_len]

        if len(chunk) < sr:
            break

        print(f"\nChunk @ {pointer/sr:.2f}s | size={chunk_size}s")

        mode = agent.decide_model()
        results = asr_agent(chunk, sr, mode)

        text, conf = selection_agent(results)

        print(f"   Mode: {mode} | Conf: {conf:.2f}")

        if agent.need_redecode(conf):
            print("   🔁 Re-decoding...")
            subs = split_audio(chunk, sr, SMALL_CHUNK)
            texts = []

            for sub in subs:
                sub_res = asr_agent(sub, sr, "both")
                t, _ = selection_agent(sub_res)
                texts.append(t)

            text = " ".join(texts)

        agent.update(conf)
        final_text += text + " "
        confidences.append(conf)

        pointer += chunk_len

    # ==============================
    # SAVE TEXT
    # ==============================
    with open(f"{file}_output.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    # ==============================
    # SPEAKER OUTPUT
    # ==============================
    speaker_lines = simple_speaker_split(final_text)

    with open(f"{file}_speaker.txt", "w", encoding="utf-8") as f:
        for line in speaker_lines:
            f.write(line + "\n")

    # ==============================
    # GRAPH
    # ==============================
    plt.plot(confidences)
    plt.title(file)
    plt.xlabel("Chunks")
    plt.ylabel("Confidence")
    plt.savefig(f"{file}_graph.png")
    plt.close()

    # ==============================
    # METRICS
    # ==============================
    evaluate(file, final_text)

    return final_text

# ==============================
# MAIN
# ==============================
files = [f for f in os.listdir(FOLDER) if f.endswith(".wav")]

for f in files[:2]:   # change later to full dataset
    process_file(f)

print("\n✅ ALL DONE")