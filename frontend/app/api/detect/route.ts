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
    const baseUrl = BACKEND_URLS.ASR.main;
    const statusRes = await axios.get(`${baseUrl}/status`, { timeout: 2000 });
    const isLoaded = statusRes.data.model_loaded === true || statusRes.data.model_state === 'loaded';
    if (isLoaded) {
      await axios.post(`${baseUrl}/toggle`, null, { timeout: 20000 });
    }
  } catch (e: any) {
    console.log(`Toggle down skipped/failed for ASR: ${e.message}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('[Detect Proxy] Starting deepfake detection');
    const formData = await request.formData();
    
    const audio = formData.get('audio');
    const modelIdsRaw = formData.get('model_ids') as string;
    const modelIds = modelIdsRaw ? modelIdsRaw.split(',') : ['CM1'];

    console.log('[Detect Proxy] Details:', {
      hasAudio: !!audio,
      modelIds,
      audioType: audio instanceof Blob ? 'Blob/File' : typeof audio
    });

    if (!audio) {
      return NextResponse.json(
        { error: 'Missing required field: audio' },
        { status: 422 }
      );
    }

    const results = [];

    // Process each model
    for (const modelId of modelIds) {
      const modelConfig = BACKEND_URLS.DETECT[modelId as keyof typeof BACKEND_URLS.DETECT];
      if (!modelConfig) {
        console.warn(`[Detect Proxy] Unknown model: ${modelId}`);
        continue;
      }

      const baseUrl = modelConfig.main;

      // Mark model as inferring
      setModelInferring(modelId, true);

      try {
        // 1. VRAM Check
        let modelOnline = true;
        try {
          const statusRes = await axios.get(`${baseUrl}/status`, { timeout: 3000 });
          if (!statusRes.data.model_loaded) {
            // Try to toggle down other models (best effort, ignore failures)
            try {
              await toggleAllDown();
            } catch (toggleErr) {
              console.log(`[Detect Proxy] Toggle all down failed, continuing anyway...`);
            }
            // Try to toggle on this model
            try {
              const params = new URLSearchParams();
              params.append('mode', '1');
              await axios.post(`${baseUrl}/toggle`, params, { timeout: 5000 });
              // Wait for model to load
              await new Promise(resolve => setTimeout(resolve, 2000));
            } catch (toggleErr: any) {
              console.log(`[Detect Proxy] Toggle on failed for ${modelId}, will try detection anyway: ${toggleErr.message}`);
            }
          }
        } catch (e: any) {
          console.log(`[Detect Proxy] Model ${modelId} is offline`);
          modelOnline = false;
        }

        // If model is offline, skip to next model
        if (!modelOnline) {
          results.push({
            model: modelId,
            confidence: 0,
            label: 'Offline'
          });
          continue;
        }

        // 2. Send to backend
        const backendFd = new FormData();
        backendFd.append('audio', audio as Blob, 'audio.wav');

        console.log(`[Detect Proxy] Forwarding to ${baseUrl}/detect...`);

        // Create a timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('TIMEOUT')), 30000); // 30 second timeout
        });

        try {
          const response = await Promise.race([
            fetch(`${baseUrl}/detect`, {
              method: 'POST',
              body: backendFd,
            }),
            timeoutPromise
          ]);

          if (!response.ok) {
            const err = await response.text();
            console.error(`[Detect Proxy] Backend Error for ${modelId}:`, err);
            results.push({
              model: modelId,
              confidence: 0,
              label: 'Error'
            });
            // Server responded with error - stop processing further models
            setModelInferring(modelId, false);
            console.log(`[Detect Proxy] Stopping model processing due to server error for ${modelId}`);
            break;
          }

          const data = await response.json();
          results.push({
            model: modelId,
            sourceBonaFide: (data.probabilities?.bonafide ?? 0) * 100,
            sourceSpoof: (data.probabilities?.spoof ?? 0) * 100,
            sourceThreshold: data.threshold ?? 50,
            confidence: data.confidence ?? 0,
            label: data.label || data.prediction || 'Unknown',
            threshold: data.threshold ?? 50
          });
        } catch (detectionError: any) {
          console.error(`[Detect Proxy] Detection error for ${modelId}:`, detectionError.message);
          
          // Check if it's a timeout error
          if (detectionError.message === 'TIMEOUT') {
            console.log(`[Detect Proxy] Timeout for ${modelId}, continuing with next model`);
            results.push({
              model: modelId,
              confidence: 0,
              label: 'Timeout'
            });
            // Continue to next model on timeout
          } else {
            console.log(`[Detect Proxy] Non-timeout error for ${modelId}, stopping processing`);
            results.push({
              model: modelId,
              confidence: 0,
              label: 'Offline'
            });
            // Stop processing on non-timeout errors (server not responding)
            setModelInferring(modelId, false);
            break;
          }
        }
      } catch (fetchError: any) {
        console.error(`[Detect Proxy] Outer fetch error for ${modelId}:`, fetchError.message);
        results.push({
          model: modelId,
          confidence: 0,
          label: 'Offline'
        });
        // Stop processing on outer errors
        setModelInferring(modelId, false);
        break;
      } finally {
        // Clear inferring state
        setModelInferring(modelId, false);
      }
    }

    return NextResponse.json({ results }, { status: 200 });

  } catch (error: any) {
    console.error('[Detect Proxy] Global Error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    );
  }
}
