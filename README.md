# Video Editor Agent

An intelligent, AI-assisted video editing platform that leverages Large Language Models to transform natural language commands into precise video manipulations. Built for speed, precision, and a seamless user experience.

---

## ✨ Key Features

- **🤖 AI-Powered Editing**: Communicate with your video via Google Gemini. Trim, cut, speed up, and remove dead air/silence through natural language, previewed as accept/reject proposals before anything is applied.
- **👁️ Hybrid Vision Pipeline**: Short clips are analyzed as real video directly; longer clips use sprite-sheet thumbnails (with extra frames at detected scene changes) plus an on-demand direct-video zoom-in when a cut needs frame-accurate precision.
- **⏱️ Professional Timeline**: A high-fidelity, interactive timeline built with custom React components, offering multi-segment manipulation, undo/redo, and frame-accurate seeking.
- **🔒 Self-Host Ready**: Shared API-key auth plus per-endpoint rate limiting, so a publicly reachable instance can't be run up in quota or disk usage.
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
- **AI Engine**: [Google Gemini](https://deepmind.google/technologies/gemini/)
- **Validation**: Pydantic v2

---

## 🏗️ Architecture Overview

The project follows a decoupled architecture designed for scale:

1.  **Frontend**: A responsive web application that manages the user interaction layer, timeline state, and real-time previews.
2.  **Backend**: A stateless API server that orchestrates video processing jobs, references an uploaded video by a persisted id instead of a server-side session, and interfaces with Gemini for intent parsing and content analysis.
3.  **Media Layer**: A structured storage system for uploads, processed outputs, and temporary sprite assets, swept on a TTL so nothing accumulates indefinitely.

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
cp .env.example .env # Set GEMINI_API_KEY; also set API_KEY before exposing this publicly
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

- `POST /analyze/sprites`: Generates sprite-sheet thumbnails and persists the source video for later calls to reference.
- `POST /agent/plan`: Plans edits from a prompt + video context (direct video, sprites, or both), returning reviewable proposals (accept/reject before anything applies).
- `POST /agent/summarize`: Rolling summary of older chat turns, so long editing sessions don't grow prompt cost unbounded.
- `POST /export/from-file`: Applies trim/speed ranges and renders the final export.

---

## 🧪 Quality Assurance

- **Backend**: Tested with `pytest` for robust endpoint validation and FFmpeg logic.
- **Frontend**: Unit tests powered by `vitest`.
- **CI/CD**: Automated GitHub Actions for linting, testing, and deployment readiness.

---

*Intentionally lean. Built for the future of creative editing.*

