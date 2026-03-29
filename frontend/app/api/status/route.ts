import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import { BACKEND_URLS } from '@/lib/backend';
import { modelState, setModelInferring, ModelStatus as IModelStatus } from '@/lib/modelState';

async function checkModelStatus(url: string, modelName: string): Promise<'Offline' | 'Model Unload' | 'Model Loaded' | 'Inferring'> {
  let actualStatus: 'Offline' | 'Model Unload' | 'Model Loaded' = 'Offline';
  
  try {
    const response = await axios.get(`${url}/status`, { timeout: 10000 });
    const isLoaded = response.data.model_loaded === true || response.data.model_state === 'loaded';
    if (isLoaded) {
      actualStatus = 'Model Loaded';
    } else {
      actualStatus = 'Model Unload';
    }
  } catch (error: any) {
    actualStatus = 'Offline';
  }
  
  if (modelState.inferringModels.has(modelName)) {
    return 'Inferring';
  }
  
  return actualStatus;
}

async function updateCachedStatus() {
  try {
    const statusChecks: Promise<IModelStatus>[] = [];

    for (const [name, config] of Object.entries(BACKEND_URLS.TTS)) {
      statusChecks.push(
        checkModelStatus(config.main, name).then(status => ({ name, status, category: 'TTS' as const }))
      );
    }

    for (const [name, config] of Object.entries(BACKEND_URLS.DETECT)) {
      statusChecks.push(
        checkModelStatus(config.main, name).then(status => ({ name, status, category: 'DETECT' as const }))
      );
    }

    statusChecks.push(
      checkModelStatus(BACKEND_URLS.ASR.main, 'ASR').then(status => ({ name: 'ASR', status, category: 'ASR' as const }))
    );

    const statuses = await Promise.all(statusChecks);
    modelState.cachedStatuses = statuses;
  } catch (error: any) {
    console.error('[Status] Error updating cache:', error);
  }
}

async function startBackgroundPolling() {
  if (modelState.interval) return;
  
  updateCachedStatus();
  modelState.interval = setInterval(async () => {
    await updateCachedStatus();
  }, 7000);
}

startBackgroundPolling();

export async function GET(request: NextRequest) {
  return NextResponse.json({ statuses: modelState.cachedStatuses });
}

export async function POST(request: NextRequest) {
  try {
    const { modelName, isInferring } = await request.json();
    
    if (modelName === '__CLEAR_ALL__') {
      modelState.inferringModels.clear();
      return NextResponse.json({ success: true, message: 'All inferring states cleared' });
    }
    
    setModelInferring(modelName, isInferring);
    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
