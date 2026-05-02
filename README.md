# 🎙️ Agent-Controlled Long-Form ASR for Meetings

## 📌 Overview

This project implements a **multi-agent Automatic Speech Recognition (ASR) system** designed for long-duration audio such as meetings and court transcripts.
The system dynamically adapts segmentation, context window size, and model selection to improve transcription accuracy while managing computational cost.

---

## 🚀 Key Features

* 🤖 **Agent-Based Control System**

  * Adaptive chunking
  * Confidence-based re-decoding
  * Dynamic model selection (Tiny vs Base)

* 🎧 **Supports Long Audio**

  * Handles 20–40 minute meeting recordings

* 🧠 **Multi-Agent Pipeline**

  * Segmentation Agent
  * ASR Agent
  * Model Selection Agent
  * Evaluation Agent

* 📊 **Performance Metrics**

  * Word Error Rate (WER)
  * Word Information Lost (WIL)
  * Word Information Preserved (WIP)

* 📈 **Visualization**

  * Confidence score graphs across chunks

* 👥 **Speaker-wise Output (Simplified)**

  * Generates structured transcripts

---

## 📂 Dataset

This project uses the **AMI Meeting Corpus (IS1000 subset)**:

* IS1000a
* IS1000b
* IS1000c (optional)

👉 Audio types:

* Headset (recommended)
* Microphone Array (for robustness testing)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Agentic-LongForm-ASR.git
cd Agentic-LongForm-ASR
```

---

### 2. Install dependencies

```bash
pip install openai-whisper
pip install librosa numpy matplotlib jiwer soundfile torch
```

---

### 3. Install FFmpeg (Required)

Download from:
https://www.gyan.dev/ffmpeg/builds/

Add to PATH:

```
C:\ffmpeg\ffmpeg-8.1-essentials_build\bin
```

Verify:

```bash
ffmpeg -version
```

---

## ▶️ Usage

Update dataset path in code:

```python
FOLDER = "path_to_your_AMI_audio"
```

Run:

```bash
python SP_endsem.py
```

---

## 📊 Output Files

For each audio file:

* `_output.txt` → Full transcription
* `_speaker.txt` → Speaker-wise transcript
* `_graph.png` → Confidence visualization

---

## 🧠 System Architecture

```
Audio Input
   ↓
[Agent Controller]
   ↓
[Adaptive Chunking]
   ↓
[ASR Models (Tiny + Base)]
   ↓
[Selection Agent]
   ↓
[Confidence-Based Re-decoding]
   ↓
[Evaluation Metrics]
   ↓
Final Transcript
```

---

## 📉 Metrics

| Metric | Description                |
| ------ | -------------------------- |
| WER    | Word Error Rate            |
| WIL    | Word Information Lost      |
| WIP    | Word Information Preserved |

---

## 🔍 Observations

* Headset audio yields higher accuracy than array recordings
* Low-confidence segments trigger re-decoding
* Adaptive chunking improves long-form transcription stability

---

## 📚 References

* ICASSP 2025: Chunk-Adaptive Transformer ASR
* ASRU 2025: Memory-Efficient Long-Context ASR
* AMI Meeting Corpus

---


## 📌 Future Work

* Real speaker diarization (WhisperX / pyannote)
* GPU optimization
* Real-time streaming ASR
* Courtroom dataset extension

---

## ⭐ Acknowledgment

This project was developed as part of an academic assignment on **Agent-Controlled Long-Form ASR Systems (2025)**.

---
