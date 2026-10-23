"use client";

import React from "react";

interface ModelControlsProps {
  n: number;
  onChange: (n: number) => void;
  modelUsed?: number;
  strategy?: string;
  latencyMs?: number;
}

export function ModelControls({
  n,
  onChange,
  modelUsed,
  strategy,
  latencyMs,
}: ModelControlsProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 py-1 text-xs">
      <div className="flex items-center gap-2">
        <span className="text-slate-400 font-semibold tracking-wide">Context Model:</span>
        <div className="inline-flex rounded-xl bg-black p-1 border border-neutral-800 shadow-inner">
          <button
            type="button"
            onClick={() => onChange(2)}
            className={`px-3.5 py-1.5 rounded-lg font-bold transition-all duration-150 cursor-pointer ${
              n === 2
                ? "bg-cyan-400 text-black shadow-[0_0_12px_rgba(0,240,255,0.6)]"
                : "text-slate-400 hover:text-cyan-300"
            }`}
          >
            Bigram (n=2)
          </button>
          <button
            type="button"
            onClick={() => onChange(3)}
            className={`px-3.5 py-1.5 rounded-lg font-bold transition-all duration-150 cursor-pointer ${
              n === 3
                ? "bg-cyan-400 text-black shadow-[0_0_12px_rgba(0,240,255,0.6)]"
                : "text-slate-400 hover:text-cyan-300"
            }`}
          >
            Trigram (n=3)
          </button>
        </div>
      </div>

      {modelUsed !== undefined && latencyMs !== undefined && latencyMs > 0 && (
        <div className="flex items-center gap-2 text-xs">
          <span className="inline-flex items-center px-2.5 py-1 rounded-lg bg-black border border-amber-500/40 text-amber-300 font-mono text-[11px] shadow-[0_0_8px_rgba(255,215,0,0.2)]">
            {modelUsed === 3 ? "Trigram active" : modelUsed === 2 ? "Bigram active" : "Unigram fallback"}
          </span>
          <span className="tabular-nums font-mono text-cyan-400 font-bold drop-shadow-[0_0_6px_rgba(0,240,255,0.4)]">
            {latencyMs.toFixed(1)}ms
          </span>
        </div>
      )}
    </div>
  );
}
