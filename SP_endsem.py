import os
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
import time
import pandas as pd
import traceback

# ==============================
# RESULTS FOLDER
# ==============================
RESULTS_FOLDER = "SP_results_final_V2"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"Results will be saved in: {RESULTS_FOLDER}")

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ==============================
# CONFIG
# ==============================

BASE_CHUNK = 45
SMALL_CHUNK = 20
LARGE_CHUNK = 60

LOW_CONF = -1.0
HIGH_CONF = -0.6

REFERENCE_TEXTS = {}

# ==============================
# LOAD MODELS
# ==============================
print("Loading models...")
tiny_model = whisper.load_model("tiny").to(device)
base_model = whisper.load_model("base").to(device)

# ==============================
# AGENT CONTROLLER
# ==============================
class Agent:
    def __init__(self):
        self.history = []
        self.prev_text = ""
        self.redecode_count = 0
        self.total_chunks = 0
        self.total_redecodes = 0
        self.start_time = time.time()
        self.low_conf_streak = 0

        self.chunk_stats = {
            SMALL_CHUNK: 0,
            BASE_CHUNK: 0,
            LARGE_CHUNK: 0
        }

        self.model_usage = {
            "tiny": 0,
            "base": 0
        }

    def avg_conf(self):
        if not self.history:
            return -1.0
        return sum(self.history) / len(self.history)

    def audio_difficulty(self, chunk, sr):

        rms = np.mean(
            librosa.feature.rms(y=chunk)
        )

        silence_ratio = np.mean(
            np.abs(chunk) < 0.005
        )

        difficulty_score = 0

        # Very weak speech
        if rms < 0.015:
            difficulty_score += 1

        # Excessive silence
        if silence_ratio > 0.70:
            difficulty_score += 1

        return difficulty_score

    def decide_chunk(self, chunk=None, sr=None):

        avg = self.avg_conf()
        difficulty = 0

        if chunk is not None:
            difficulty = self.audio_difficulty(
                chunk,
                sr
            )

        # Hard audio
        if avg < -1.4 and difficulty >= 1:
            chunk_size = SMALL_CHUNK

        # Easy audio
        elif avg > -0.5 and difficulty == 0:
            chunk_size = LARGE_CHUNK

        else:
            chunk_size = BASE_CHUNK

        self.chunk_stats[chunk_size] += 1

        return chunk_size

    def decide_model(self, chunk=None, sr=None):

        avg = self.avg_conf()

        difficulty = 0

        budget_mode = self.compute_budget_mode()

        if chunk is not None:
            difficulty = self.audio_difficulty(
                chunk,
                sr
            )

        # Compute-saving mode
        if budget_mode:

            if avg < -1.5 and difficulty >= 1:
                self.model_usage["base"] += 1
                return "base"

            self.model_usage["tiny"] += 1
            return "tiny"

        # Easy chunk
        if avg > -0.7 and difficulty == 0:
            self.model_usage["tiny"] += 1
            return "tiny"

        # Hard chunk
        self.model_usage["base"] += 1
        return "base"

    def need_redecode(self, conf):
        if self.redecode_count > 5:
            return False
        if conf < LOW_CONF:
            self.redecode_count += 1
            self.total_redecodes += 1
            return True
        return False

    def get_context(self):

        avg = self.avg_conf()

        # High confidence → larger context
        if avg > -0.5:
            context_size = 250

        # Low confidence → shorter context
        elif avg < -1:
            context_size = 80

        else:
            context_size = 150

        return self.prev_text[-context_size:]

    def update(self, conf, text):
        self.history.append(conf)
        if len(self.history) > 5:
            self.history.pop(0)
        self.prev_text += " " + text
        self.total_chunks += 1

    # ==============================
    # CONTEXT REFRESH AGENT
    # ==============================
    def refresh_context(self, conf):

        if conf < -1.3:
            self.low_conf_streak += 1
        else:
            self.low_conf_streak = 0

        # Two consecutive bad chunks
        if self.low_conf_streak >= 2:
            print("⚠ Context Refreshed")

            self.prev_text = ""

            self.low_conf_streak = 0

    # ==============================
    # COMPUTE BUDGET AGENT
    # ==============================
    def compute_budget_mode(self):

        runtime = time.time() - self.start_time

        # Runtime budget exceeded
        if runtime > 120:
            return True

        return False

    def compute_metrics(self):
        runtime = time.time() - self.start_time
        return {
            "chunks": self.total_chunks,
            "redecodes": self.total_redecodes,
            "runtime": runtime,
            "redecode_ratio": self.total_redecodes / max(self.total_chunks,1),
            "tiny_ratio": self.model_usage["tiny"] /
                          max(self.total_chunks, 1),

            "base_ratio": self.model_usage["base"] /
                          max(self.total_chunks, 1)
        }

# ==============================
# ASR AGENT
# ==============================
def asr_agent(chunk, sr, mode, context):

    sf.write("temp.wav", chunk, sr)

    if mode == "tiny":
        return [
            tiny_model.transcribe(
                "temp.wav",
                initial_prompt=context,
                fp16=(device == "cuda")
            )
        ]

    elif mode == "base":
        return [
            base_model.transcribe(
                "temp.wav",
                initial_prompt=context,
                fp16=(device == "cuda")
            )
        ]
# ==============================
# CONFIDENCE
# ==============================
def get_conf(res):
    if res.get("segments"):
        return np.mean([s["avg_logprob"] for s in res["segments"]])
    return -2.0

def select_best(results):
    best_text, best_conf = "", -2.0
    for r in results:
        c = get_conf(r)
        if c > best_conf:
            best_conf = c
            best_text = r["text"]
    return best_text, best_conf

# ==============================
# PROCESS FILE
# ==============================
def process_file(file):
    print(f"\nProcessing: {file}")
    audio_type = "Headset" if "Headset" in file else "Array"

    agent = Agent()
    path = file
    file = os.path.basename(file)
    audio, sr = librosa.load(path, sr=16000, mono=True)

    pointer = 0
    final_text = ""
    confidences = []
    model_history = []
    chunk_history = []
    confidence_gains = []

    while pointer < len(audio):

        preview = audio[
                  pointer:pointer + int(BASE_CHUNK * sr)
                  ]

        chunk_size = agent.decide_chunk(
            preview,
            sr
        )
        chunk_len = int(chunk_size * sr)
        chunk = audio[pointer:pointer+chunk_len]

        if len(chunk) < sr:
            break

        context = agent.get_context()
        mode = agent.decide_model(
            chunk,
            sr
        )
        print("Running model:", mode)
        results = asr_agent(chunk, sr, mode, context)
        text, conf = select_best(results)
        # ==============================
        # DECISION LOGGING
        # ==============================
        redecode_flag = agent.need_redecode(conf)

        print(
            f"\n--------------------------------\n"
            f"Chunk Number : {agent.total_chunks + 1}\n"
            f"Confidence   : {round(conf, 3)}\n"
            f"Chunk Size   : {chunk_size}s\n"
            f"Selected Model : {mode}\n"
            f"Re-decode Triggered : {redecode_flag}\n"
            f"--------------------------------"
        )

        if redecode_flag:

            before_conf = conf

            subs = [
                chunk[i:i + int(SMALL_CHUNK * sr)]
                for i in range(
                    0,
                    len(chunk),
                    int(SMALL_CHUNK * sr)
                )
            ]

            texts = []
            sub_confs = []

            for s in subs:
                sub_res = asr_agent(
                    s,
                    sr,
                    "base",
                    context
                )

                t, c = select_best(sub_res)

                texts.append(t)
                sub_confs.append(c)

            text = " ".join(texts)

            after_conf = np.mean(sub_confs)

            confidence_gains.append(
                after_conf - before_conf
            )

            conf = after_conf
        confidences.append(conf)

        model_history.append(mode)

        chunk_history.append(chunk_size)

        final_text += " " + text

        agent.refresh_context(conf)
        agent.update(conf, text)

        pointer += chunk_len

    # ==============================
    # SAVE OUTPUT
    # ==============================
    output_path = os.path.join(
        RESULTS_FOLDER,
        f"{file}_output.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    # ==============================
    # GRAPH
    # ==============================
    plt.figure()
    plt.plot(confidences)
    plt.title(file)
    plt.xlabel("Chunks")
    plt.ylabel("Confidence")
    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            f"{file}_graph.png"
        )
    )
    plt.close()

    # ==============================
    # DECISION TIMELINE GRAPH
    # ==============================
    plt.figure(figsize=(10, 5))

    x = range(len(confidences))

    plt.plot(x, confidences, label="Confidence")

    for i in range(len(model_history)):
        if model_history[i] == "tiny":
            plt.scatter(i, confidences[i], marker='o', label="Tiny" if i == 0 else "")
        else:
            plt.scatter(i, confidences[i], marker='x', label="Base" if i == 0 else "")

    plt.xlabel("Chunk Number")
    plt.ylabel("Confidence")
    plt.title(f"{file} Decision Timeline")
    plt.legend()
    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            f"{file}_timeline.png"
        )
    )
    plt.close()

    metrics = agent.compute_metrics()
    metrics["small_chunks"] = agent.chunk_stats[SMALL_CHUNK]
    metrics["base_chunks"] = agent.chunk_stats[BASE_CHUNK]
    metrics["large_chunks"] = agent.chunk_stats[LARGE_CHUNK]
    metrics["tiny_usage"] = agent.model_usage["tiny"]
    metrics["base_usage"] = agent.model_usage["base"]
    metrics["audio_type"] = audio_type
    metrics["tiny_selected"] = agent.model_usage["tiny"]
    metrics["base_selected"] = agent.model_usage["base"]
    metrics["small_chunk_used"] = agent.chunk_stats[SMALL_CHUNK]
    metrics["base_chunk_used"] = agent.chunk_stats[BASE_CHUNK]
    metrics["large_chunk_used"] = agent.chunk_stats[LARGE_CHUNK]
    metrics["avg_confidence"] = (
        np.mean(confidences)
        if len(confidences) > 0
        else 0
    )


    # ==============================
    # CONFIDENCE HEATMAP
    # ==============================
    plt.figure(figsize=(10, 2))
    heat = np.array(confidences).reshape(1, -1)
    plt.imshow(
        heat,
        aspect='auto',
        cmap='coolwarm'
    )
    plt.colorbar(label="Confidence")
    plt.yticks([])
    plt.xlabel("Chunks")
    plt.title(f"{file} Confidence Heatmap")
    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            f"{file}_heatmap.png"
        )
    )
    plt.close()
    metrics["avg_conf_gain"] = (
        np.mean(confidence_gains)
        if confidence_gains
        else 0
    )
    return metrics

# ==============================
# MAIN
# ==============================

TEST_FOLDER = r"C:\Users\joshe\OneDrive - Amrita vishwa vidyapeetham\_collegefiles\SEM_files_\Sem_4\SP\project_endsem\amicorpus\test_cases"

files = [
    os.path.join(TEST_FOLDER, f)
    for f in os.listdir(TEST_FOLDER)
    if f.endswith(".wav")
]

print("\nSelected Files:")
for f in files:
    print(os.path.basename(f))

all_results = []

for f in files:
    try:
        res = process_file(f)
        res["file"] = os.path.basename(f)
        all_results.append(res)

    except Exception as e:

        print("ERROR:", f)

        traceback.print_exc()

# ==============================
# TABLE OUTPUT
# ==============================
if len(all_results) == 0:
    print("No files processed successfully.")
    exit()
df = pd.DataFrame(all_results)
df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "agent_results.csv"
    ),
    index=False
)

# ==============================
# AGENT DECISION SUMMARY TABLE
# ==============================
summary = pd.DataFrame({
    "Metric": [
        "Tiny Selected",
        "Base Selected",
        "Small Chunks",
        "Base Chunks",
        "Large Chunks"
    ],
    "Count": [
        df["tiny_selected"].sum(),
        df["base_selected"].sum(),
        df["small_chunk_used"].sum(),
        df["base_chunk_used"].sum(),
        df["large_chunk_used"].sum()
    ]
})

summary.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "agent_decision_summary.csv"
    ),
    index=False
)

print("\n📊 Agent Decision Summary")
print(summary)

print("\n📊 Agent-wise Results Table:")
print(df)

# ==============================
# OVERALL METRICS
# ==============================
overall = {
    "avg_runtime": df["runtime"].mean(),
    "avg_chunks": df["chunks"].mean(),
    "avg_redecodes": df["redecodes"].mean(),
    "avg_redecode_ratio": df["redecode_ratio"].mean(),
    "avg_confidence": df["avg_confidence"].mean(),
    "avg_conf_gain": df["avg_conf_gain"].mean(),
    "total_conf_improvement": df["avg_conf_gain"].sum()
}

print("\n📊 Overall Performance:")
for k, v in overall.items():
    print(k, ":", round(v, 3))

# ==============================
# COMPARISON GRAPH
# ==============================
plt.figure()
plt.bar(df["file"], df["runtime"])
plt.xticks(rotation=45)
plt.title("Runtime Comparison")
plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "runtime_comparison.png"
    )
)
plt.close()

plt.figure()
plt.bar(df["file"], df["redecode_ratio"])
plt.xticks(rotation=45)
plt.title("Re-decode Ratio Comparison")
plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "redecode_comparison.png"
    )
)
plt.close()

plt.figure()
grouped = df.groupby("audio_type")["runtime"].mean()
plt.bar(grouped.index, grouped.values)
plt.title("Headset vs Array Runtime")
plt.ylabel("Average Runtime")
plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "headset_vs_array_runtime.png"
    )
)
plt.close()

chunk_totals = {
    "Small": df["small_chunks"].sum(),
    "Base": df["base_chunks"].sum(),
    "Large": df["large_chunks"].sum()
}
plt.figure()
plt.bar(chunk_totals.keys(), chunk_totals.values())
plt.title("Chunk Distribution")
plt.ylabel("Count")
plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "chunk_distribution.png"
    )
)
plt.close()

model_totals = {
    "Tiny": df["tiny_usage"].sum(),
    "Base": df["base_usage"].sum()
}
plt.figure()
plt.bar(model_totals.keys(), model_totals.values())
plt.title("Tiny vs Base Model Usage")
plt.ylabel("Usage Count")
plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "model_usage.png"
    )
)
plt.close()
# ==============================
# CONFIDENCE VS COMPUTE
# ==============================

plt.figure(figsize=(8,5))

plt.scatter(
    df["runtime"],
    df["avg_confidence"]
)
for i, txt in enumerate(df["file"]):
    plt.annotate(
        txt,
        (
            df["runtime"].iloc[i],
            df["avg_confidence"].iloc[i]
        )
    )
plt.xlabel("Runtime (seconds)")
plt.ylabel("Average Confidence")

plt.title(
    "Confidence vs Runtime Trade-off"
)

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "confidence_vs_runtime.png"
    )
)

plt.close()

print("\n✅ EVERYTHING COMPLETE")

