# Real vs. Deepfake Analysis Application

A full-stack application for recording voice, generating deepfakes, and comparing detection results.

## Architecture

- **Backend**: FastAPI gateway managing Gradio worker processes
- **Frontend**: Next.js 14 with App Router, TypeScript, Tailwind CSS
- **Cold Start**: Models load on-demand to conserve VRAM

## Setup

### Backend

```bash
docker compose up -d
```

API will be available at `http://localhost:8000`

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

## Usage

1. **Record Real Audio**: Click "Start Recording" to capture your voice
2. **Generate Deepfake**: Enter text and click "Generate Deepfake"
3. **Analyze Both**: Click "Analyze Both Audios" to compare detection results

## Features

- 🎤 Real-time audio recording with MediaRecorder API
- 🤖 Dynamic TTS model selection (TTS V1, TTS V2)
- 🔍 Multi-model deepfake detection (CM V1, CM V2)
- 📊 Side-by-side comparison with visual scoring
- 🚀 Cold start architecture (models start/stop on-demand)
- 💾 VRAM-safe sequential processing

## API Endpoints

### POST /generate-tts
Generate TTS audio from text

**Request:**
- `text`: Text to synthesize
- `model_id`: TTS model name (e.g., "TTS V1")
- `reference_audio`: Optional reference audio file

**Response:** Audio file (WAV)

### POST /detect
Detect deepfakes in audio

**Request:**
- `audio`: Audio file to analyze
- `model_ids`: Comma-separated model names (e.g., "CM V1,CM V2")

**Response:**
```json
{
  "results": [
    {
      "model": "CM V1",
      "confidence": 85.5,
      "label": "Deepfake"
    }
  ]
}
```

## Technology Stack

### Backend
- FastAPI
- Gradio Client
- psutil (process management)
- Python subprocess

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Axios
- Lucide React (icons)

## Development

The application follows a strict resource management pattern:
1. Controller never loads models directly
2. Workers start → predict → stop for each request
3. Sequential processing prevents VRAM overflow
4. Process cleanup on shutdown

## Notes

- Cold start may take 5-10 seconds per model
- Models run sequentially to conserve resources
- Audio recording requires microphone permissions
- CORS enabled for localhost:3000
