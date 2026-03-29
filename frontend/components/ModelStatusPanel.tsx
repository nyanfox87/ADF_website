'use client';


import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle, XCircle, Loader } from 'lucide-react';
import { toggleALLDown } from '@/lib/api';
import { BACKEND_URLS } from '@/lib/backend';

export interface ModelStatus {
  name: string;
  status: 'Offline' | 'Model Unload' | 'Model Loaded' | 'Inferring';
  category: string;
}

export default function ModelStatusPanel() {
  const [statuses, setStatuses] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [closeLoading, setCloseLoading] = useState(false);

  const fetchStatus = async () => {
    console.log('[ModelStatusPanel] Fetching status...');
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      console.log('[ModelStatusPanel] Status data:', data);
      setStatuses(data.statuses || []);
    } catch (error) {
      console.error('[ModelStatusPanel] Failed to fetch model statuses:', error);
    } finally {
      setLoading(false);
    }
  };

  // Build the full model list from BACKEND_URLS
  const allModels: ModelStatus[] = React.useMemo(() => {
    const models: ModelStatus[] = [];
    for (const [name] of Object.entries(BACKEND_URLS.TTS)) {
      models.push({ name, status: 'Offline', category: 'TTS' });
    }
    for (const [name] of Object.entries(BACKEND_URLS.DETECT)) {
      models.push({ name, status: 'Offline', category: 'DETECT' });
    }
    models.push({ name: 'ASR', status: 'Offline', category: 'ASR' });
    return models;
  }, []);

  useEffect(() => {
    console.log('[ModelStatusPanel] Component mounted, starting status polling');
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Refresh every 5 seconds
    console.log('[ModelStatusPanel] Polling interval started');
    return () => {
      console.log('[ModelStatusPanel] Component unmounting, clearing interval');
      clearInterval(interval);
    };
  }, []);

  const handleCloseAll = async () => {
    setCloseLoading(true);
    try {
      await toggleALLDown();
      alert('All models closed!');
      await fetchStatus(); // Refresh status after closing
    } catch (e) {
      alert('Failed to close all models');
    } finally {
      setCloseLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Model Loaded': return 'text-green-400';
      case 'Inferring': return 'text-blue-400';
      case 'Model Unload': return 'text-yellow-400';
      case 'Offline': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Model Loaded': return <CheckCircle className="w-4 h-4" />;
      case 'Inferring': return <Loader className="w-4 h-4 animate-spin" />;
      case 'Model Unload': return <Activity className="w-4 h-4" />;
      case 'Offline': return <XCircle className="w-4 h-4" />;
      default: return null;
    }
  };

  // Merge backend statuses with allModels to ensure all are shown
  const mergedStatuses = allModels.map((model) => {
    const found = statuses.find((s) => s.name === model.name && s.category === model.category);
    return found ? found : model;
  });

  const groupedStatuses = mergedStatuses.reduce((acc, status) => {
    if (!acc[status.category]) acc[status.category] = [];
    acc[status.category].push(status);
    return acc;
  }, {} as Record<string, ModelStatus[]>);

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-3">
        <Activity className="w-5 h-5 text-cyan-400" />
        <h2 className="text-lg font-bold text-white uppercase tracking-wide">Model Status</h2>
        {loading && <Loader className="w-4 h-4 text-slate-500 animate-spin ml-auto" />}
      </div>

      {/* Close All Button */}
      <button
        onClick={handleCloseAll}
        disabled={closeLoading}
        className="w-full mb-4 flex items-center justify-center gap-2 px-4 py-2.5 bg-red-700 hover:bg-red-600 text-white rounded-lg shadow text-xs font-bold uppercase tracking-wider disabled:opacity-50 transition-colors"
      >
        <XCircle className="w-4 h-4" />
        {closeLoading ? 'Closing...' : 'Close All Models'}
      </button>

      <div className="space-y-4">
        {Object.entries(groupedStatuses).map(([category, models]) => (
          <div key={category}>
            <h3 className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-2">{category}</h3>
            <div className="space-y-2">
              {models.map((model) => (
                <div 
                  key={model.name}
                  className="flex items-center justify-between bg-black/40 rounded-lg border border-slate-800 px-3 py-2 hover:border-slate-700 transition-colors"
                >
                  <span className="text-sm font-medium text-slate-300">{model.name}</span>
                  <div className={`flex items-center gap-2 text-xs font-mono ${getStatusColor(model.status)}`}>
                    {getStatusIcon(model.status)}
                    <span className="uppercase tracking-wider">{model.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Never show 'No models found' - always show all models, even if all are offline */}
      </div>
    </div>
  );
}
