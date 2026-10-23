"use client";

import React from "react";
import { SuggestionItem } from "@/lib/types";

interface SuggestionChipProps {
  index: number;
  item: SuggestionItem;
  onSelect: (word: string) => void;
}

export function SuggestionChip({ index, item, onSelect }: SuggestionChipProps) {
  const percentage = Math.max(0, Math.min(100, Math.round(item.probability * 100)));

  return (
    <button
      type="button"
      onClick={() => onSelect(item.word)}
      className="group relative flex items-center justify-between w-full px-4 py-3.5 rounded-2xl bg-black/80 hover:bg-neutral-950 border border-neutral-800 hover:border-cyan-400 transition-all duration-200 text-left overflow-hidden shadow-md hover:shadow-[0_0_20px_rgba(0,240,255,0.25)] cursor-pointer"
    >
      {/* Background progress tint */}
      <div
        className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500/15 to-amber-500/10 group-hover:from-cyan-500/25 group-hover:to-amber-500/15 transition-all duration-300 pointer-events-none rounded-2xl"
        style={{ width: `${percentage}%` }}
      />

      {/* Word & Rank Badge */}
      <div className="relative z-10 flex items-center gap-3.5 min-w-0">
        <span className="flex-shrink-0 flex items-center justify-center w-7 h-7 rounded-lg bg-black text-amber-400 group-hover:text-black group-hover:bg-amber-400 font-mono text-xs font-bold border border-amber-500/50 shadow-[0_0_8px_rgba(255,215,0,0.3)] transition-all duration-150">
          {index + 1}
        </span>
        <span
          className="text-xl font-bold text-slate-100 group-hover:text-cyan-300 drop-shadow-[0_0_6px_rgba(0,240,255,0.3)] truncate tracking-wide"
          style={{ fontFamily: "var(--font-ethiopic), system-ui, sans-serif" }}
        >
          {item.word}
        </span>
      </div>

      {/* Probability info */}
      <div className="relative z-10 flex items-center gap-3 flex-shrink-0 text-xs">
        <div className="hidden sm:block w-24 h-2 bg-neutral-900 rounded-full overflow-hidden border border-neutral-800">
          <div
            className="h-full bg-gradient-to-r from-cyan-400 via-sky-300 to-amber-400 rounded-full transition-all duration-300 shadow-[0_0_8px_rgba(0,240,255,0.5)]"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className="font-mono text-amber-300 font-bold tabular-nums text-xs drop-shadow-[0_0_5px_rgba(255,215,0,0.3)]">
          {(item.probability * 100).toFixed(1)}%
        </span>
      </div>
    </button>
  );
}
