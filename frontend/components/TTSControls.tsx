'use client';

import React, { useState } from 'react';
import { Sparkles, Languages } from 'lucide-react';
import { generateTTS, transcribeAudio } from '@/lib/api';
import { BACKEND_URLS } from '@/lib/backend';

export default function TTSControls({ onGenerate, isDisabled, realAudio }: { onGenerate: (b: Blob, model: string) => void, isDisabled: boolean, realAudio: Blob | null }) {
  const ttsModels = Object.keys(BACKEND_URLS.TTS);
  const [loading, setLoading] = useState(false);
  const [asrLoading, setAsrLoading] = useState(false);
  const [model, setModel] = useState(ttsModels[0] || 'IndexTTS2');
  const [text, setText] = useState('');

  const handleTranscribe = async () => {
    if (!realAudio) return;
    setAsrLoading(true);
    try {
      const transcribedText = await transcribeAudio(realAudio);
      setText(transcribedText);
    } catch (e) {
      console.error(e);
      alert('ASR Failed. Make sure backend /asr is running.');
    } finally {
      setAsrLoading(false);
    }
  };

  const handleClick = async () => {
    if (!text.trim()) {
      alert('Please enter text to generate TTS');
      return;
    }
    // Allow generation without ref audio
    setLoading(true);
    try {
      const mockBlob = await generateTTS(text, model, realAudio ?? new Blob([], { type: 'audio/wav' }));
      onGenerate(mockBlob, model);
      // Don't clear text - keep it for next generation
    } catch (e) { 
      console.error(e);
      alert('Failed to generate TTS. Make sure the backend is running.');
    } 
    finally { setLoading(false); }
  };

  return (
    <div className="flex flex-col items-center gap-3 w-full md:w-48">
      <div className="relative w-full bg-slate-900 border border-slate-700 p-4 rounded-xl shadow-xl flex flex-col gap-3">
        {/* ASR Button */}
        <button
          onClick={handleTranscribe}
          disabled={!realAudio || asrLoading}
          className="w-full py-2 bg-cyan-900/50 hover:bg-cyan-800 border border-cyan-700/50 text-cyan-400 text-[10px] font-bold uppercase tracking-widest rounded transition-all flex items-center justify-center gap-2 group disabled:opacity-30"
        >
          {asrLoading ? (
            <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <>
              <Languages className="w-3 h-3 group-hover:rotate-12 transition-transform" />
              Auto Transcribe (ASR)
            </>
          )}
        </button>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to generate TTS..."
          disabled={loading || isDisabled}
          className="w-full bg-black border border-slate-700 text-xs text-slate-300 p-2 rounded outline-none focus:border-cyan-500 font-mono placeholder-slate-600 disabled:opacity-50 resize-none h-16"
        />

        {/* Dropdown from sketch */}
        <select 
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={loading || isDisabled}
          className="w-full bg-black border border-slate-700 text-xs text-slate-300 p-2 rounded outline-none focus:border-cyan-500 font-mono disabled:opacity-50"
        >
          {ttsModels.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <button
          onClick={handleClick}
          disabled={loading || isDisabled}
          className="w-full py-3 bg-fuchsia-700 hover:bg-fuchsia-600 disabled:bg-slate-800 disabled:text-slate-600 text-white text-xs font-bold uppercase tracking-wider rounded transition-colors flex items-center justify-center gap-2"
        >
           {loading ? (
             <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
           ) : (
             <>
               Generate Deepfake <Sparkles className="w-3 h-3" />
             </>
           )}
        </button>
      </div>
    </div>
  );
}