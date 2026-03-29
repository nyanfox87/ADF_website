import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import { BACKEND_URLS } from '@/lib/backend';

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
    const baseUrl = BACKEND_URLS.ASR.main;
    const statusRes = await axios.get(`${baseUrl}/status`, { timeout: 2000 });
    const isLoaded = statusRes.data.model_loaded === true || statusRes.data.model_state === 'loaded';
    if (isLoaded) {
      await axios.post(`${baseUrl}/toggle`, null, { timeout: 3000 });
    }
  } catch (e: any) {
    console.log(`Toggle down skipped/failed for ASR: ${e.message}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('[Proxy] Content-Type:', request.headers.get('content-type'));
    const formData = await request.formData();
    
    // Log all keys received to debug
    const allKeys = Array.from(formData.keys());
    console.log('[Proxy] Incoming keys:', allKeys);

    if (allKeys.length === 0) {
       console.log('[Proxy] CAUTION: FormData is empty!');
    }

    const modelIdRaw = formData.get('model_id') as string;
    const modelId = (modelIdRaw && modelIdRaw in BACKEND_URLS.TTS ? modelIdRaw : 'IndexTTS2') as keyof typeof BACKEND_URLS.TTS;
    const text = formData.get('text');
    const refAudio = formData.get('ref_audio');
    // Use TTS model URLs from shared backend config
    const baseUrl = BACKEND_URLS.TTS[modelId].main;

    console.log('[Proxy] Details:', { 
        modelId, 
        hasText: !!text, 
        hasRefAudio: !!refAudio,
        refAudioType: refAudio instanceof Blob ? 'Blob/File' : typeof refAudio 
    });

    if (!text || !refAudio) {
        const missing = [];
        if (!text) missing.push('text');
        if (!refAudio) missing.push('ref_audio');
        
        console.error('[Proxy] 422 Error - Missing fields:', missing);
        return NextResponse.json(
            { error: `Missing required fields: ${missing.join(', ')}`, receivedKeys: allKeys }, 
            { status: 422 }
        );
    }

    // 1. VRAM Check
    try {
      const statusRes = await axios.get(`${baseUrl}/status`, { timeout: 3000 });
      if (!statusRes.data.model_loaded) {
        await toggleAllDown();
        const params = new URLSearchParams();
        params.append('mode', '1');
        await axios.post(`${baseUrl}/toggle`, params);
      }
    } catch (e) {
      console.log('[Proxy] Status/Toggle failed, ignoring...');
    }

    // 2. Prepare Backend Request
    const backendFd = new FormData();
    backendFd.append('text', text as string);

    // Always append ref_audio, even if empty, to satisfy backend schema
    if (refAudio) {
      const fileName = (refAudio as any).name || 'input.wav';
      backendFd.append('ref_audio', refAudio as Blob, fileName);
    } else {
      // Send an empty wav Blob if missing
      const emptyBlob = new Blob([], { type: 'audio/wav' });
      backendFd.append('ref_audio', emptyBlob, 'input.wav');
    }

    console.log(`[Proxy] Forwarding to ${baseUrl}/tts...`);
    
    const response = await fetch(`${baseUrl}/tts`, {
      method: 'POST',
      body: backendFd,
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('[Proxy] Backend Error:', err);
      return NextResponse.json({ error: 'Backend error', details: err }, { status: response.status });
    }

    const buffer = await response.arrayBuffer();
    return new NextResponse(buffer, {
      status: 200,
      headers: { 'Content-Type': 'audio/wav' }
    });

  } catch (error: any) {
    console.error('[Proxy] Global Error:', error);
    return NextResponse.json({ error: 'Internal server error', details: error.message }, { status: 500 });
  }
}
