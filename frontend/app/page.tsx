'use client';

import React, { useState, useEffect } from 'react';
import AudioRecorder from '@/components/AudioRecorder';
import TTSControls from '@/components/TTSControls';
import ConfidenceMatrix from '@/components/ConfidenceMatrix';
import ModelStatusPanel from '@/components/ModelStatusPanel';
import { detectDeepfake, DetectionResult, toggleALLDown, clearAllInferringStates } from '@/lib/api';
import { BACKEND_URLS } from '@/lib/backend';
import { ShieldCheck, ArrowRight, UploadCloud, Activity, XCircle, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react';


export default function Home() {
  const [realAudio, setRealAudio] = useState<Blob | null>(null);
  const [fakeAudio, setFakeAudio] = useState<Blob | null>(null);
  const [realAudioUrl, setRealAudioUrl] = useState<string>('');
  const [fakeAudioUrl, setFakeAudioUrl] = useState<string>('');
  const [fakeAudioModel, setFakeAudioModel] = useState<string>('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Get available DETECT models from backend configuration
  const models = Object.keys(BACKEND_URLS.DETECT);
  
  const [results, setResults] = useState<{
    model: string;
    sourceConfidence?: number;
    fakeConfidence?: number;
    label?: string;
  }[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Clear any stuck inferring states on mount
  useEffect(() => {
    clearAllInferringStates();
  }, []);

  const handleRealAudioInput = (blob: Blob) => {
    setRealAudio(blob);
    if (realAudioUrl) URL.revokeObjectURL(realAudioUrl);
    setRealAudioUrl(URL.createObjectURL(blob));
    setResults([]);
  };

  const handleFakeAudioGenerated = (blob: Blob, model: string) => {
    setFakeAudio(blob);
    if (fakeAudioUrl) URL.revokeObjectURL(fakeAudioUrl);
    setFakeAudioUrl(URL.createObjectURL(blob));
    setFakeAudioModel(model);
    setResults([]);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleRealAudioInput(file);
    }
  };

  const handleFakeFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFakeAudioGenerated(file, 'IndexTTS2');
    }
  };

  const handleAnalyze = async () => {
    if (!realAudio) return;
    setIsAnalyzing(true);
    setResults([]);

    try {
      // Process models sequentially to avoid VRAM race conditions
      // (each model toggles all others down before loading itself)
      for (const modelId of models) {
        try {
          // Step 1: Analyze Real Audio
          const sourceResp = await detectDeepfake(realAudio, modelId);
          const sRes = sourceResp.results[0] || { confidence: 0, label: 'Offline' };

          // sourceConfidence = spoof probability (0–100): higher means more fake
          const sourceConfidence = sRes.sourceSpoof;
          const sourceLabel = sRes.label !== 'BonaFide' && sRes.label !== 'Spoof' ? sRes.label : undefined;

          // Add partial result (Real only)
          setResults(prev => {
            const filtered = prev.filter(r => r.model !== modelId);
            return [...filtered, { model: modelId, sourceConfidence, label: sourceLabel }];
          });

          // Step 2: Analyze Fake Audio (if exists) with the same loaded model
          let fakeConfidence: number | undefined = undefined;

          if (fakeAudio) {
            const fakeResp = await detectDeepfake(fakeAudio, modelId);
            const fRes = fakeResp.results[0];

            if (fRes && fRes.label !== 'Offline' && fRes.label !== 'Error') {
              fakeConfidence = fRes.sourceSpoof;
            }
          }

          // Update with full result (Real + Fake)
          setResults(prev => {
            const filtered = prev.filter(r => r.model !== modelId);
            return [...filtered, { model: modelId, sourceConfidence, fakeConfidence, label: sourceLabel }];
          });

        } catch (error) {
          console.error(`Analysis failed for model ${modelId}:`, error);
          setResults(prev => {
            const filtered = prev.filter(r => r.model !== modelId);
            return [...filtered, { model: modelId, label: 'Error' }];
          });
        }
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Clear real audio handler
  const handleClearAudio = () => {
    setRealAudio(null);
    if (realAudioUrl) URL.revokeObjectURL(realAudioUrl);
    setRealAudioUrl('');
    setResults([]);
  };

  // Clear fake audio handler
  const handleClearFakeAudio = () => {
    setFakeAudio(null);
    if (fakeAudioUrl) URL.revokeObjectURL(fakeAudioUrl);
    setFakeAudioUrl('');
    setFakeAudioModel('');
  };

  // Use synthetic audio as input
  const handleUseSyntheticAsInput = () => {
    if (!fakeAudio) return;
    handleRealAudioInput(fakeAudio);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-12 font-sans selection:bg-cyan-500/30">
      <div className="flex gap-6 max-w-[1600px] mx-auto">
        {/* Main Content */}
        <div className="flex-1">
          <div className="space-y-8">
            
            {/* ===== HEADER ===== */}
            <header className="flex items-center gap-3 border-b border-slate-800 pb-6">
              <div className="p-2 bg-cyan-950/50 rounded-lg border border-cyan-800">
                <ShieldCheck className="w-6 h-6 text-cyan-400" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white uppercase">Audio Deepfake Detection</h1>
            </header>

            {/* ===== TOP SECTION: RECORD -> GENERATE -> OUTPUT ===== */}
            {/* This matches the 3-part flow in your drawing */}
            <section className="grid md:grid-cols-[1fr_auto_1fr] gap-6 items-stretch">
              {/* 1. LEFT: RECORD YOUR VOICE */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between relative group">
                <h2 className="text-lg font-bold text-white mb-1">Speech Input</h2>
                {/* <p className="text-xs text-slate-500 font-mono mb-4">RECORD OR UPLOAD</p> */}
                
                <AudioRecorder 
                  onRecordingComplete={handleRealAudioInput} 
                  audioUrl={realAudioUrl}
                  onClear={handleClearAudio}
                  onFileUpload={handleFileUpload}
                />
              </div>

              {/* 2. MIDDLE: ARROW & CONTROLS (The "TTS V / Generate" block) */}
              <div className="flex flex-col justify-center items-center gap-4 py-4 md:px-2">
                <div className="hidden md:block w-px h-12 bg-gradient-to-b from-transparent via-slate-700 to-transparent"></div>
                
                {/* The Control Box */}
                <TTSControls 
                  onGenerate={handleFakeAudioGenerated}
                  isDisabled={!realAudio}
                  realAudio={realAudio}
                />

                <div className="hidden md:block w-px h-12 bg-gradient-to-b from-transparent via-slate-700 to-transparent"></div>
              </div>

              {/* 3. RIGHT: GENERATED WAVEFORM */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between relative group">
                <h2 className="text-lg font-bold text-white mb-1">Speech Synthetic</h2>
                
                {fakeAudioUrl ? (
                  <div className="space-y-3">
                    <AudioRecorder 
                      onRecordingComplete={() => {}} 
                      audioUrl={fakeAudioUrl}
                      onClear={handleClearFakeAudio}
                      hideRecord={true}
                      hideUpload={true}
                      hideReRecord={true}
                    />
                    <div className="flex-1 py-4 bg-slate-800/40 border border-fuchsia-700/40 rounded-2xl text-fuchsia-300 text-xs font-black uppercase tracking-widest flex items-center justify-center gap-2">
                      <Sparkles size={18} /> {fakeAudioModel || 'IndexTTS2'}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full min-h-[200px] gap-4 text-slate-600">
                    <Sparkles className="w-8 h-8" />
                    <span className="text-xs font-mono uppercase">No Synthetic Audio</span>
                    <button
                      onClick={() => document.getElementById('fake-audio-upload')?.click()}
                      className="w-full py-4 bg-slate-800/40 hover:bg-slate-800 border border-slate-700 rounded-2xl text-slate-300 text-xs font-black uppercase tracking-widest flex items-center justify-center gap-2 transition-all hover:text-fuchsia-400 hover:border-fuchsia-400/30"
                    >
                      <UploadCloud size={18} /> Upload .WAV
                    </button>
                    <input type="file" id="fake-audio-upload" className="hidden" accept="audio/*" onChange={handleFakeFileUpload} />
                  </div>
                )}
              </div>
            </section>

            {/* ===== MIDDLE: DETECT BUTTON ===== */}
            <section>
              <button
              onClick={handleAnalyze}
              disabled={!realAudio || isAnalyzing}
              className="w-full group relative h-16 bg-gradient-to-r from-slate-700 via-cyan-700 via-40% via-blue-600 via-60% to-slate-700 hover:from-slate-600 hover:via-cyan-500 hover:via-40% hover:via-blue-500 hover:via-60% hover:to-slate-600 border-2 border-cyan-400 hover:border-cyan-300 overflow-hidden rounded-2xl shadow-[0_0_30px_rgba(6,182,212,0.3)] hover:shadow-[0_0_50px_rgba(6,182,212,0.5)] transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed disabled:shadow-none"
              >
              <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(6,182,212,0.2)_50%,transparent_75%)] bg-[length:250%_250%] group-hover:animate-[shimmer_2s_linear_infinite]"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/20 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000"></div>
              <div className="relative flex items-center justify-center gap-4">
                {isAnalyzing ? (
                <span className="font-mono text-cyan-50 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)] animate-pulse tracking-[0.2em] text-lg font-bold">PROCESSING...</span>
                ) : (
                <>
                  <span className="text-2xl font-black text-cyan-50 drop-shadow-[0_0_8px_rgba(6,182,212,0.5)] tracking-[0.2em] uppercase transition-all">DETECT</span>
                  <Activity className="w-7 h-7 text-cyan-50 drop-shadow-[0_0_8px_rgba(6,182,212,0.5)] transition-all" />
                </>
                )}
              </div>
              </button>
            </section>

            {/* ===== BOTTOM: RESULTS TABLE (Matching Sketch: CM1, CM2...) ===== */}
            <section className="animate-in slide-in-from-bottom-4 fade-in duration-700">
              <h3 className="text-xl font-mono text-slate-400 uppercase mb-4 border-l-4 border-cyan-500 pl-4">Performance</h3>
              <ConfidenceMatrix 
                models={models} 
                results={results} 
                isAnalyzing={isAnalyzing} 
                showFake={true}
              />
            </section>
          </div>
        </div>

        {/* Collapsible Right Sidebar - Model Status */}
        <div className="relative flex items-start">
           {/* Toggle Button */}
           <button 
             onClick={() => setIsSidebarOpen(!isSidebarOpen)}
             className={`sticky top-12 z-20 w-8 h-8 flex items-center justify-center bg-slate-900 border border-slate-700 rounded-full text-slate-400 hover:text-cyan-400 hover:border-cyan-500/50 shadow-[0_0_20px_rgba(0,0,0,0.5)] transition-all duration-500 ${isSidebarOpen ? '-mr-4' : 'mr-0'}`}
             title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
           >
             {isSidebarOpen ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
           </button>

           <aside 
             className={`transition-all duration-500 ease-in-out shrink-0 ${isSidebarOpen ? 'w-80 opacity-100 ml-4' : 'w-0 opacity-0 ml-0 overflow-hidden'}`}
           >
             <div className="w-80">
               <ModelStatusPanel />
             </div>
           </aside>
        </div>
      </div>
    </main>
  );
}