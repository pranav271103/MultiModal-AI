# Multimodal AI Framework — Audio, Video, Image, and Text Understanding

This repository contains modular code for a multimodal AI project built to handle four different types of human-like data inputs: Audio, Video, Images, and Text. The solution uses open-source tools such as LangChain, BLIP, CLIP, YOLO, ResNet, Wav2Vec2, and Ollama models for semantic analysis.

---

## 💡 System Requirements
- Python 3.9 or higher  
- GPU recommended (NVIDIA CUDA supported)  
- Tested on Windows 11 and Google Colab  

---

## ⚙️ Ollama Model Installation

Before running any notebook, install and configure **Ollama** to enable local LLM usage.

**For Windows:**  
1. Download Ollama installer from: [https://ollama.com/download](https://ollama.com/download)  
2. Install and open `Ollama` on your machine.  
3. Pull the required model:  
```bash
ollama pull granite3-dense:2b
```
4. Make sure Ollama is running in the background.

---

## 🧑‍💻 Notebook Overview

### 🎙️ 1. AudioPart.ipynb  
This notebook converts audio files (MP3/WAV) into text using Google Speech Recognition and transcribes the audio into meaningful responses with the help of Ollama's Granite3-Dense LLM.

**How to Run:**
- Install dependencies:
```bash
pip install speechrecognition pydub langchain langchain_ollama
```
- Place your `.mp3` or `.wav` file in the project directory.
- Update the file path in `audio_path` variable.
- Run the notebook cell by cell.

---

### 🎬 2. VideoEncoding.ipynb  
This notebook extracts frames from a video, generates captions using the BLIP model, transcribes audio using Wav2Vec2, and summarizes the combined information via Ollama LLM.

**How to Run:**
- Install dependencies:
```bash
pip install opencv-python-headless torch torchvision transformers tqdm librosa langchain_ollama ollama
```
- Update the `VIDEO_PATH` in the notebook.
- Make sure FFmpeg is installed for audio extraction (`sudo apt-get install ffmpeg` for Linux, or download for Windows).
- Run the notebook for a complete video-based AI summary.

---

### 🔼️ 3. ImagePart.ipynb  
Processes images to detect objects using YOLOv8, generates semantic labels using CLIP, and creates a descriptive summary using Ollama's LLM.

**How to Run:**
- Install dependencies:
```bash
pip install ultralytics torch torchvision openai-clip langchain_ollama
```
- Replace `image_path` with your target image path.
- Run the notebook to get object detection, labels, and natural language summary.

---

### 📚 4. CombinedAi.ipynb (Text Part)  
Processes text documents like PDF files, splits them into chunks, stores them in a vector database, and uses Ollama LLM for semantic Q&A and web-augmented search.

**How to Run:**
- Install dependencies:
```bash
pip install langchain langchain_community langchain_ollama chromadb tavily-client
```
- Replace `file_path` with your PDF document's path.
- Run the notebook for document query answering and Tavily web search.

---

## 🏁 Final Note

This repo is modular and designed for clear demonstration of multimodal pipelines — you can run each notebook independently depending on the data type you want to process.

---

Happy experimenting! 🚀
