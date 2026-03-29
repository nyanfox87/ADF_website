import React from 'react';
import { AlertCircle } from 'lucide-react';
import { Loader2 } from 'lucide-react';

const THRESHOLD = 50;

interface ResultRow {
  model: string;
  sourceConfidence?: number;  // 0–100, where 100 = 100% fake (spoof probability)
  fakeConfidence?: number;    // 0–100, where 100 = 100% fake (spoof probability)
  label?: string;
}

export default function ConfidenceMatrix({
  models,
  results,
  isAnalyzing,
  showFake = false
}: {
  models: string[];
  results: ResultRow[];
  isAnalyzing?: boolean;
  showFake?: boolean;
}) {
  const hasFake = showFake || results.some(r => r.fakeConfidence !== undefined);

  const resultMap: Record<string, ResultRow> = {};
  results.forEach(r => {
    resultMap[r.model] = r;
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Table Header */}
      <div className={`grid ${hasFake ? 'grid-cols-7' : 'grid-cols-4'} bg-slate-950 border-b border-slate-800 p-4 text-xs font-mono text-slate-300 uppercase tracking-widest`}>
        <div className="col-span-1">Model ID</div>
        <div className="col-span-3 text-center border-l border-slate-800/50">
          <span className="text-cyan-500">Speech Input</span>
        </div>
        {hasFake && (
          <div className="col-span-3 text-center border-l border-slate-800/50">
            <span className="text-fuchsia-500">Speech Synthetic</span>
          </div>
        )}
      </div>

      <div className={`grid ${hasFake ? 'grid-cols-7' : 'grid-cols-4'} bg-slate-900/50 border-b border-slate-800 p-2 text-xs font-mono text-slate-400 uppercase`}>
        <div></div>
        <div className="text-center">Threshold</div>
        <div className="text-center">Confidence</div>
        <div className="text-center">Category</div>
        {hasFake && (
          <>
            <div className="text-center border-l border-slate-800/50">Threshold</div>
            <div className="text-center">Confidence</div>
            <div className="text-center">Category</div>
          </>
        )}
      </div>

      <div className="divide-y divide-slate-800">
        {models.map((model) => {
          const row = resultMap[model];

          // Case 1: No data yet
          if (!row) {
            return (
              <div key={model} className={`grid ${hasFake ? 'grid-cols-7' : 'grid-cols-4'} p-4 items-center bg-slate-900/30`}>
                <div className="col-span-1 border-r border-slate-800/50">
                  <span className="font-bold text-white font-mono text-sm">{model}</span>
                </div>
                {isAnalyzing ? (
                  <div className={`${hasFake ? 'col-span-6' : 'col-span-3'} text-center flex items-center justify-center gap-2 animate-pulse`}>
                    <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                    <span className="text-sm font-mono text-cyan-400 uppercase">Inferring...</span>
                  </div>
                ) : (
                  <div className={`${hasFake ? 'col-span-6' : 'col-span-3'} text-center`}>
                    <span className="text-lg font-mono text-slate-600">-</span>
                  </div>
                )}
              </div>
            );
          }

          // Case 2: Offline or Error
          if (row.label === 'Offline' || row.label === 'Error') {
            return (
              <div key={model} className={`grid ${hasFake ? 'grid-cols-7' : 'grid-cols-4'} p-4 items-center bg-slate-900/10`}>
                <div className="col-span-1 border-r border-slate-800/50">
                  <span className="font-bold text-white font-mono text-sm">{row.model}</span>
                </div>
                <div className={`${hasFake ? 'col-span-6' : 'col-span-3'} text-center flex items-center justify-center gap-2`}>
                  <AlertCircle className="w-4 h-4 text-red-500" />
                  <span className="text-sm font-mono text-red-500 uppercase">{row.label}</span>
                </div>
              </div>
            );
          }

          // Case 3: Results
          const sourceIsFake = (row.sourceConfidence ?? 0) > THRESHOLD;
          const fakeIsFake = (row.fakeConfidence ?? 0) > THRESHOLD;

          return (
            <div key={model} className={`grid ${hasFake ? 'grid-cols-7' : 'grid-cols-4'} p-4 items-center hover:bg-slate-800/30 transition-colors`}>
              {/* Model Name */}
              <div className="col-span-1 border-r border-slate-800/50">
                <span className="font-bold text-white font-mono text-sm">{row.model}</span>
                <div className="text-[8px] text-slate-500">NEURAL ENGINE</div>
              </div>

              {/* Source: Threshold */}
              <div className="text-center px-2">
                <span className="text-sm font-mono text-yellow-500">{THRESHOLD.toFixed(2)}</span>
              </div>
              {/* Source: Confidence */}
              <div className="text-center px-2">
                <span className={`text-md font-mono font-bold ${sourceIsFake ? 'text-red-400' : 'text-cyan-400'}`}>
                  {row.sourceConfidence?.toFixed(2) ?? '0.00'}
                </span>
              </div>
              {/* Source: Category */}
              <div className="text-center px-2 border-r border-slate-800/50">
                <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                  sourceIsFake
                    ? 'bg-red-900/40 text-red-400 border border-red-700/50'
                    : 'bg-cyan-900/40 text-cyan-400 border border-cyan-700/50'
                }`}>
                  {sourceIsFake ? 'FAKE' : 'BONAFIDE'}
                </span>
              </div>

              {/* Fake Results */}
              {hasFake && (
                <>
                  {row.fakeConfidence !== undefined ? (
                    <>
                      {/* Fake: Threshold */}
                      <div className="text-center px-2">
                        <span className="text-sm font-mono text-yellow-500">{THRESHOLD.toFixed(2)}</span>
                      </div>
                      {/* Fake: Confidence */}
                      <div className="text-center px-2">
                        <span className={`text-md font-mono font-bold ${fakeIsFake ? 'text-red-400' : 'text-cyan-400'}`}>
                          {row.fakeConfidence.toFixed(2)}
                        </span>
                      </div>
                      {/* Fake: Category */}
                      <div className="text-center px-2">
                        <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                          fakeIsFake
                            ? 'bg-red-900/40 text-red-400 border border-red-700/50'
                            : 'bg-cyan-900/40 text-cyan-400 border border-cyan-700/50'
                        }`}>
                          {fakeIsFake ? 'FAKE' : 'BONAFIDE'}
                        </span>
                      </div>
                    </>
                  ) : isAnalyzing ? (
                    <div className="col-span-3 text-center flex items-center justify-center gap-2 animate-pulse">
                      <Loader2 className="w-3 h-3 text-fuchsia-400 animate-spin" />
                      <span className="text-[10px] font-mono text-fuchsia-400 uppercase">Analyzing...</span>
                    </div>
                  ) : (
                    <div className="col-span-3 text-center">
                      <span className="text-lg font-mono text-slate-600">-</span>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
