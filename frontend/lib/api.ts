import axios from 'axios';

// Inactivity timer logic
let lastApiCall = Date.now();
let inactivityTimer: ReturnType<typeof setTimeout> | null = null;

async function checkBackendStatusAndMaybeClose() {
  try {
    // Only close if no API call in last 30 minutes
    if (Date.now() - lastApiCall >= 30 * 60 * 1000) {
      await fetch('/api/close-all', { method: 'POST' });
      // Optionally, you can log or notify here
      console.log('Backends closed due to inactivity.');
    }
  } catch (e) {
    // Ignore errors
  } finally {
    // Schedule next check
    inactivityTimer = setTimeout(checkBackendStatusAndMaybeClose, 30 * 60 * 1000);
  }
}

function resetInactivityTimer() {
  lastApiCall = Date.now();
  if (inactivityTimer) clearTimeout(inactivityTimer);
  inactivityTimer = setTimeout(checkBackendStatusAndMaybeClose, 30 * 60 * 1000);
}

// Start the inactivity timer on module load
resetInactivityTimer();

// Helper to update model inferring state
export const setModelInferring = async (modelName: string, isInferring: boolean) => {
  try {
    await fetch('/api/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelName, isInferring })
    });
  } catch (e) {
    console.error('Failed to update inferring state:', e);
  }
};

// Helper to clear all inferring states (useful for debugging/recovery)
export const clearAllInferringStates = async () => {
  try {
    await fetch('/api/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelName: '__CLEAR_ALL__', isInferring: false })
    });
    console.log('[API] Cleared all inferring states');
  } catch (e) {
    console.error('Failed to clear inferring states:', e);
  }
};

export interface DetectionResult {
  model: string;
  confidence: number;
  label: string;
  threshold?: number;
  sourceBonaFide?: number;
  sourceSpoof?: number;
}

export interface DetectionResponse {
  results: DetectionResult[];
}

export const generateTTS = async (
  text: string,
  modelId: string,
  referenceAudio?: Blob | File // allow undefined
): Promise<Blob> => {
  // If no referenceAudio, send empty Blob
  const audio = referenceAudio ?? new Blob([], { type: 'audio/wav' });

  const fd = new FormData();
  fd.append('text', text);
  fd.append('model_id', modelId);
  fd.append('ref_audio', audio, 'reference.wav');

  // Debug: verify FormData content
  const keys: string[] = [];
  fd.forEach((_, key) => keys.push(key));
  
  console.log('[Frontend] Sending TTS request:', {
    text,
    modelId,
    blobSize: audio.size,
    blobType: audio.type,
    formDataKeys: keys,
    isBlob: audio instanceof Blob
  });

  // Mark model as inferring
  await setModelInferring(modelId, true);

  // Remove toggleALLDown() - let backend TTS route handle VRAM management

  try {
    // Add timeout to prevent hanging forever
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    
    const response = await fetch('/api/tts', {
      method: 'POST',
      body: fd,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail;
      try {
          const json = JSON.parse(errorText);
          errorDetail = json.details || json.error || errorText;
      } catch {
          errorDetail = errorText;
      }
      throw new Error(`Server Error (${response.status}): ${errorDetail}`);
    }

    return await response.blob();
  } catch (error) {
    console.error('[Frontend] TTS Error:', error);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('TTS request timed out after 60 seconds');
    }
    throw error;
  } finally {
    // Clear inferring state - always execute
    console.log(`[Frontend] Clearing inferring state for ${modelId}`);
    await setModelInferring(modelId, false);
  }
};

export const detectDeepfake = async (
  audio: Blob,
  modelId: string
): Promise<DetectionResponse> => {
  const formData = new FormData();
  formData.append('audio', audio, 'audio.wav');
  formData.append('model_ids', modelId);

  console.log('[Frontend] Sending detection request:', {
    modelId,
    audioSize: audio.size,
    audioType: audio.type
  });

  // Remove toggleALLDown() - let backend detect route handle VRAM management

  const response = await axios.post<DetectionResponse>('/api/detect', formData);
  return response.data;
};

export const transcribeAudio = async (audio: Blob): Promise<string> => {
  const formData = new FormData();
  formData.append('audio', audio, 'audio.wav');

  try {
    const response = await axios.post<{ text: string }>('/api/asr', formData);
    return response.data.text;
  } catch (error) {
    console.error('Transcription error:', error);
    return "ASR Error";
  }
};

export const toggleALLDown = async (): Promise<void> => {
  await fetch('/api/close-all', { method: 'POST' });
};
