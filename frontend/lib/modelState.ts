import { BACKEND_URLS } from './backend';

export interface ModelStatus {
  name: string;
  status: 'Offline' | 'Model Unload' | 'Model Loaded' | 'Inferring';
  category: string;
}

// Use global object to persist state across hot reloads (dev mode)
const g = globalThis as any;
if (!g.__MODEL_STATUS_STATE__) {
  // Initialize with all models as Offline so we don't return an empty list on first request
  const initialStatuses: ModelStatus[] = [];
  for (const [name] of Object.entries(BACKEND_URLS.TTS)) {
    initialStatuses.push({ name, status: 'Offline', category: 'TTS' });
  }
  for (const [name] of Object.entries(BACKEND_URLS.DETECT)) {
    initialStatuses.push({ name, status: 'Offline', category: 'DETECT' });
  }
  initialStatuses.push({ name: 'ASR', status: 'Offline', category: 'ASR' });

  g.__MODEL_STATUS_STATE__ = {
    inferringModels: new Set<string>(),
    cachedStatuses: initialStatuses,
    isPolling: false,
    interval: null,
  };
}

export const modelState = g.__MODEL_STATUS_STATE__;

export function setModelInferring(modelName: string, isInferring: boolean) {
  if (isInferring) {
    console.log(`[Status] Setting ${modelName} to Inferring`);
    modelState.inferringModels.add(modelName);
  } else {
    console.log(`[Status] Clearing Inferring state for ${modelName}`);
    modelState.inferringModels.delete(modelName);
  }
  
  // Update only the specific model in cache immediately
  const modelIndex = modelState.cachedStatuses.findIndex((s: ModelStatus) => s.name === modelName);
  if (modelIndex !== -1) {
    if (isInferring) {
      modelState.cachedStatuses[modelIndex].status = 'Inferring';
    }
  }
}
