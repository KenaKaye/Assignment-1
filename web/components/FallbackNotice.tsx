"use client";

import React from "react";
import { Info } from "lucide-react";

interface FallbackNoticeProps {
  fallback: boolean;
  strategy?: string;
  modelUsed: number;
  requestedN: number;
}

export function FallbackNotice({
  fallback,
  strategy,
  modelUsed,
  requestedN,
}: FallbackNoticeProps) {
  if (!fallback) return null;

  return (
    <div className="flex items-start gap-3 p-3.5 rounded-2xl bg-black border border-amber-500/50 text-amber-200 text-xs shadow-[0_0_15px_rgba(255,215,0,0.15)] animate-in fade-in duration-200">
      <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5 drop-shadow-[0_0_5px_rgba(255,215,0,0.6)]" />
      <div>
        <span className="font-bold text-amber-300">የሞዴል ቅነሳ ተጠቅሟል (Backoff Applied): </span>
        <span className="text-slate-300">
          {requestedN === 3 && modelUsed === 2
            ? "ለዚህ ባለ ሁለት ቃል አውድ ቀጥተኛ ባለ 3-ግራም ባለመገኘቱ ወደ ባለ 2-ግራም (Bigram) ወርዷል። (Trigram context backed off to Bigram)."
            : "አውዱ ያልተለመደ በመሆኑ ወደ ተደጋጋሚ ቃላት (Unigram fallback) ወርዷል። (Unseen context used Unigram fallback)."}
        </span>
        {strategy && (
          <span className="block mt-1 font-mono text-[11px] text-cyan-400">
            Strategy: {strategy}
          </span>
        )}
      </div>
    </div>
  );
}
