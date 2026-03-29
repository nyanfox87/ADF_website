import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import { BACKEND_URLS } from '@/lib/backend';
import { setModelInferring } from '@/lib/modelState';

async function toggleAllDown() {
  // Toggle down TTS models
  for (const [name, config] of Object.entries(BACKEND_URLS.TTS)) {
    try {
      const params = new URLSearchParams();
      params.append('mode', '0');
      await axios.post(`${config.main}/toggle`, params, { timeout: 20000 });
    } catch (e: any) {
      console.log(`Toggle down skipped/failed for TTS ${name}: ${e.message}`);
    }
  }
  
  // Toggle down DETECT models
  for (const [name, config] of Object.entries(BACKEND_URLS.DETECT)) {
    try {
      const params = new URLSearchParams();
      params.append('mode', '0');
      await axios.post(`${config.main}/toggle`, params, { timeout: 20000 });
    } catch (e: any) {
      console.log(`Toggle down skipped/failed for DETECT ${name}: ${e.message}`);
    }
  }

  // Toggle down ASR model if it's currently loaded
  try {
    const statusRes = await axios.get(`${BACKEND_URLS.ASR.main}/status`, { timeout: 2000 });
    const isLoaded = statusRes.data.model_loaded === true || statusRes.data.model_state === 'loaded';
    if (isLoaded) {
      await axios.post(`${BACKEND_URLS.ASR.main}/toggle`, null, { timeout: 20000 });
    }
  } catch (e: any) {
    console.log(`Toggle down skipped/failed for ASR: ${e.message}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('[ASR Proxy] Starting transcription');
    const formData = await request.formData();
    const audio = formData.get('audio');

    if (!audio) {
      return NextResponse.json(
        { error: 'Missing required field: audio' },
        { status: 422 }
      );
    }

    const baseUrl = BACKEND_URLS.ASR.main;
    const transcribeUrl = BACKEND_URLS.ASR.transcribe;

    // Mark model as inferring
    setModelInferring('ASR', true);

    try {
      // 1. VRAM Check & Model Load
      try {
        const statusRes = await axios.get(`${baseUrl}/status`, { timeout: 3000 });
        const isLoaded = statusRes.data.model_loaded === true || statusRes.data.model_state === 'loaded';
        if (!isLoaded) {
          // Free up memory
          await toggleAllDown();
          
          // Toggle on this model
          const params = new URLSearchParams();
          params.append('mode', '1');
          await axios.post(`${baseUrl}/toggle`, params, { timeout: 5000 });
          
          // Short wait for model to load
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (e: any) {
        console.log(`[ASR Proxy] Status/Toggle check failed: ${e.message}`);
      }

      // 2. Prepare and Forward Request
      const backendFd = new FormData();
      // Backend expects 'file' not 'audio'
      backendFd.append('file', audio as Blob, 'audio.wav');

      console.log(`[ASR Proxy] Forwarding to ${transcribeUrl}...`);

      const response = await fetch(transcribeUrl, {
        method: 'POST',
        body: backendFd,
      });

      if (!response.ok) {
        const err = await response.text();
        console.error(`[ASR Proxy] Backend Error:`, err);
        return NextResponse.json({ error: 'Backend error', details: err }, { status: response.status });
      }

      const data = await response.json();
      console.log('[ASR Proxy] Success:', data);
      
      // Expected backend response: { text: string, language: string }
      return NextResponse.json(data, { status: 200 });

    } finally {
      // Clear inferring state
      setModelInferring('ASR', false);
    }

  } catch (error: any) {
    console.error('[ASR Proxy] Global Error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    );
  }
}
