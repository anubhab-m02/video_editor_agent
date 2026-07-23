# Video Editor Agent

An intelligent, AI-assisted video editing platform that leverages Large Language Models to transform natural language commands into precise video manipulations. Built for speed, precision, and a seamless user experience.

---

## ✨ Key Features

- **🤖 AI-Powered Editing**: Communicate with your video via Google Gemini. Trimming, cutting, and speed adjustments are handled through natural language intent parsing.
- **⏱️ Professional Timeline**: A high-fidelity, interactive timeline built with custom React components, offering multi-segment manipulation and frame-accurate seeking.
- **🖼️ Intelligent Sprite Engine**: Low-latency video scrubbing powered by an automated sprite-sheet generation engine, optimizing for both performance and AI context.
- **📉 AI Insight Engine**: Smart cut suggestions from prompt + timeline context, with sprite-sheet visual analysis in progress.
- **⚡ High-Performance Backend**: A robust FastAPI backend integrated with FFmpeg for seamless, reliable, and multi-threaded video processing.

## 🛠️ Tech Stack

### Frontend
- **Framework**: [Next.js 15](https://nextjs.org/) (App Router)
- **State Management**: [Zustand](https://github.com/pmndrs/zustand)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **UI Components**: [Radix UI](https://www.radix-ui.com/) & [Lucide Icons](https://lucide.dev/)
- **Aesthetics**: Premium Glassmorphism & Framer Motion animations

### Backend
- **Core**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+)
- **Processing**: [FFmpeg](https://ffmpeg.org/) & `ffprobe`
- **AI Engine**: [Google Gemini Pro](https://deepmind.google/technologies/gemini/)
- **Validation**: Pydantic v2

---

## 🏗️ Architecture Overview

The project follows a decoupled architecture designed for scale:

1.  **Frontend**: A responsive web application that manages the user interaction layer, timeline state, and real-time previews.
2.  **Backend**: A stateless API server that orchestrates video processing jobs, manages session-based media uploads, and interfaces with Gemini for intent parsing and content analysis.
3.  **Media Layer**: A structured storage system for uploads, processed outputs, and temporary sprite assets.

---

## 🚀 Getting Started

### Prerequisites
- Node.js 20+
- Python 3.11+
- FFmpeg installed in system `PATH`

### 1. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env # Set GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```
Visit `http://localhost:3000` to start editing.

---

## 🧬 API Highlights

- `POST /analyze/sprites`: Generates tile-based previews for the UI scrubber.
- `POST /ai/suggest-cuts-from-sprites`: Returns AI-driven timestamp suggestions based on user goals and video metadata.
- `POST /export/from-file`: Applies trim/speed ranges and renders the final export.

---

## 🧪 Quality Assurance

- **Backend**: Tested with `pytest` for robust endpoint validation and FFmpeg logic.
- **Frontend**: Unit tests powered by `vitest` and `@testing-library/react`.
- **CI/CD**: Automated GitHub Actions for linting, testing, and deployment readiness.

---

*Intentionally lean. Built for the future of creative editing.*

