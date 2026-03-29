import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import { BACKEND_URLS } from '@/lib/backend';

export async function POST(request: NextRequest) {
  const results: Record<string, any> = {};
  // Flatten all model URLs for TTS, DETECT, and ASR
  const allModels = [
    ...Object.entries(BACKEND_URLS.TTS).map(([name, obj]) => [name, obj.main]),
    ...Object.entries(BACKEND_URLS.DETECT).map(([name, obj]) => [name, obj.main]),
    ['ASR', BACKEND_URLS.ASR.main],
  ];
  for (const [name, url] of allModels) {
    try {
      // Special handling for ASR which is a simple toggle without mode=0/1 support
      if (name === 'ASR') {
        const statusRes = await axios.get(`${url}/status`, { timeout: 2000 });
        const isLoaded = statusRes.data.model_loaded === true || statusRes.data.model_state === 'loaded';
        if (isLoaded) {
          await axios.post(`${url}/toggle`, null, { timeout: 20000 });
        }
        results[name] = { success: true, status: 'toggled off' };
        continue;
      }

      const params = new URLSearchParams();
      params.append('mode', '0');
      const res = await axios.post(`${url}/toggle`, params, { timeout: 20000 });
      results[name] = { success: true, status: res.status };
    } catch (e: any) {
      results[name] = { success: false, error: e.message };
    }
  }
  return NextResponse.json({ results });
}
