'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Mic, Square, X, UploadCloud, Play, Pause, RefreshCw } from 'lucide-react';

export default function AudioRecorder({ 
  onRecordingComplete, 
  audioUrl, 
  onClear, 
  onFileUpload,
  hideRecord = false,
  hideUpload = false,
  hideReRecord = false
}: { 
  onRecordingComplete: (b: Blob) => void, 
  audioUrl?: string, 
  onClear?: () => void, 
  onFileUpload?: (e: React.ChangeEvent<HTMLInputElement>) => void,
  hideRecord?: boolean,
  hideUpload?: boolean,
  hideReRecord?: boolean
}) {
  const [recording, setRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [realTimeLevel, setRealTimeLevel] = useState<number[]>(Array(20).fill(0.05));

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);
  const animationFrameRef = useRef<number>();
  const isRecordingRef = useRef(false); // 用於動畫迴圈的同步抓取

  // --- 1. 修復：即時錄音動畫渲染 ---
  const startRealTimeAnalysis = async (stream: MediaStream) => {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioContextClass();
    
    // 確保 Context 是啟動狀態
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 128; // 增加細節
    analyser.smoothingTimeConstant = 0.4; // 讓跳動更滑順
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const updateLevel = () => {
      if (!isRecordingRef.current) {
        cancelAnimationFrame(animationFrameRef.current!);
        return;
      }
      
      analyser.getByteFrequencyData(dataArray);
      // 只取中低頻部分，通常人聲在這邊跳動最明顯
      const levels = Array.from(dataArray.slice(2, 22)).map(v => (v / 255));
      setRealTimeLevel(levels);
      
      animationFrameRef.current = requestAnimationFrame(updateLevel);
    };

    updateLevel();
  };

  // --- 2. 錄音控制 ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        onRecordingComplete(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      isRecordingRef.current = true;
      mediaRecorder.current.start();
      setRecording(true);
      startRealTimeAnalysis(stream);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && recording) {
      isRecordingRef.current = false;
      mediaRecorder.current.stop();
      setRecording(false);
      setRealTimeLevel(Array(20).fill(0.05)); // 重置動畫
    }
  };

  // --- 3. 修復：進度條/波形點擊跳轉 (Seeking) ---
  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left; // 點擊位置
    const percentage = x / rect.width; // 點擊比例
    
    audioRef.current.currentTime = percentage * duration;
    setCurrentTime(audioRef.current.currentTime);
  };

  // --- 4. 波形生成與音頻監聽 ---
  const generateWaveform = useCallback(async (url: string) => {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const rawData = audioBuffer.getChannelData(0);
      const samples = 30;
      const blockSize = Math.floor(rawData.length / samples);
      const filteredData = [];
      for (let i = 0; i < samples; i++) {
        let blockStart = blockSize * i;
        let sum = 0;
        for (let j = 0; j < blockSize; j++) sum += Math.abs(rawData[blockStart + j]);
        filteredData.push(sum / blockSize);
      }
      const max = Math.max(...filteredData);
      setWaveformData(filteredData.map(n => n / max));
    } catch (err) {
      console.error('Waveform generation error:', err);
    }
  }, []);

  useEffect(() => {
    if (audioUrl) generateWaveform(audioUrl);
  }, [audioUrl, generateWaveform]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const updateProgress = () => setCurrentTime(audio.currentTime);
    const setAudioDuration = () => setDuration(audio.duration);
    const onEnded = () => { setIsPlaying(false); setCurrentTime(0); };
    
    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('loadedmetadata', setAudioDuration);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('timeupdate', updateProgress);
      audio.removeEventListener('loadedmetadata', setAudioDuration);
      audio.removeEventListener('ended', onEnded);
    };
  }, [audioUrl]);

  return (
    <div className="relative w-full max-w-2xl mx-auto space-y-3">
      <div className={`relative min-h-[160px] bg-slate-900/80 backdrop-blur-xl border-2 border-dashed rounded-3xl transition-all duration-500 flex flex-col items-center justify-center p-6 
        ${recording ? 'border-red-500 ring-4 ring-red-500/10' : 'border-slate-700 hover:border-slate-500'}`}
      >
        {audioUrl && !recording ? (
          <div className="w-full space-y-6 animate-in fade-in slide-in-from-top-4">
            <button onClick={onClear} className="absolute right-4 top-4 p-2 hover:bg-white/10 rounded-full transition-colors z-20">
              <X className="w-5 h-5 text-slate-400" />
            </button>

            {/* 互動式波形：點擊可調整進度 */}
            <div 
              className="relative h-24 flex items-center justify-center gap-[3px] px-4 cursor-pointer group/wave"
              onClick={handleSeek}
            >
               {waveformData.map((amplitude, i) => {
                 const progress = currentTime / duration;
                 const isPassed = (i / waveformData.length) < progress;
                 return (
                   <div 
                     key={i}
                     className={`w-1.5 rounded-full transition-colors duration-300 ${isPassed ? 'bg-cyan-400' : 'bg-slate-700 group-hover/wave:bg-slate-600'}`}
                     style={{ height: `${Math.max(amplitude * 100, 15)}%` }}
                   />
                 );
               })}
               {/* 懸浮掃描線 (Optional) */}
               <div className="absolute top-0 bottom-0 w-px bg-white/20 opacity-0 group-hover/wave:opacity-100 pointer-events-none" />
            </div>

            <div className="flex items-center gap-4 bg-black/20 p-4 rounded-2xl">
              <button 
                onClick={() => isPlaying ? audioRef.current?.pause() : audioRef.current?.play()}
                className="p-4 bg-cyan-500 hover:bg-cyan-400 text-black rounded-full shadow-lg shadow-cyan-500/20 transition-transform active:scale-90"
              >
                {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" className="ml-1" />}
              </button>
              
              <div className="flex-1 space-y-2">
                 <div className="flex justify-between text-xs font-mono text-slate-400">
                   <span className="text-cyan-400">{formatTime(currentTime)}</span>
                   <span>{formatTime(duration)}</span>
                 </div>
                 {/* 底部的 Progress Bar 也可以點擊 */}
                 <div 
                   className="h-2 w-full bg-slate-800 rounded-full overflow-hidden cursor-pointer"
                   onClick={handleSeek}
                 >
                    <div 
                      className="h-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]"
                      style={{ width: `${(currentTime / (duration || 1)) * 100}%` }}
                    />
                 </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-8 w-full">
            {recording ? (
              <div className="flex items-center justify-center gap-1.5 h-16 w-full">
                {realTimeLevel.map((level, i) => (
                  <div 
                    key={i} 
                    className="w-2 bg-red-500 rounded-full transition-all duration-75 ease-out"
                    style={{ 
                      height: `${Math.max(level * 100, 10)}%`,
                      opacity: 0.3 + (level * 0.7)
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="relative">
                <div className="p-6 bg-slate-800 rounded-full text-slate-500 border border-slate-700 shadow-inner">
                  <Mic size={40} />
                </div>
              </div>
            )}

            {!hideRecord && (
              <button 
                onClick={recording ? stopRecording : startRecording}
                className={`group px-10 py-4 rounded-2xl font-black text-lg tracking-widest transition-all duration-300 flex items-center gap-3 shadow-2xl ${
                  recording 
                  ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse shadow-red-500/20' 
                  : 'bg-white hover:bg-cyan-50 text-black shadow-white/5'
                }`}
              >
                {recording ? (
                  <><Square size={20} fill="currentColor" className="animate-pulse" /> STOP</>
                ) : (
                  <><Mic size={20} className="group-hover:scale-110 transition-transform" /> RECORD</>
                )}
              </button>
            )}
            
            {hideRecord && !recording && !audioUrl && (
              <span className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                Waiting for generation...
              </span>
            )}
          </div>
        )}

        <audio ref={audioRef} src={audioUrl} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} className="hidden" />
      </div>

      {(!hideUpload || (!hideReRecord && audioUrl && !recording)) && (
        <div className="flex gap-3">
          {!hideUpload && (
            <button 
              onClick={() => document.getElementById('audio-upload')?.click()}
              className="flex-1 py-4 bg-slate-800/40 hover:bg-slate-800 border border-slate-700 rounded-2xl text-slate-300 text-xs font-black uppercase tracking-widest flex items-center justify-center gap-2 transition-all hover:text-cyan-400 hover:border-cyan-400/30"
            >
              <UploadCloud size={18} /> Upload .WAV
            </button>
          )}
          {audioUrl && !hideReRecord && !recording && (
            <button 
              onClick={startRecording}
              className="px-8 py-4 bg-slate-800/40 hover:bg-slate-800 border border-slate-700 rounded-2xl text-slate-300 text-xs font-black uppercase tracking-widest flex items-center justify-center gap-2 transition-all hover:text-red-400"
            >
              <RefreshCw size={18} /> Re-record
            </button>
          )}
        </div>
      )}

      <input type="file" id="audio-upload" className="hidden" accept="audio/*" onChange={onFileUpload} />
    </div>
  );
}

function formatTime(seconds: number) {
  if (isNaN(seconds)) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}