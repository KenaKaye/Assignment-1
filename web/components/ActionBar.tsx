"use client";

import React, { useState } from "react";
import { RotateCcw, Trash2, Copy, Check } from "lucide-react";

interface ActionBarProps {
  canUndo: boolean;
  onUndo: () => void;
  onReset: () => void;
  sentence: string;
}

export function ActionBar({ canUndo, onUndo, onReset, sentence }: ActionBarProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!sentence.trim()) return;
    try {
      await navigator.clipboard.writeText(sentence.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const hasContent = sentence.trim().length > 0;

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-neutral-900">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onUndo}
          disabled={!canUndo}
          className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
            canUndo
              ? "bg-black border-cyan-500/40 text-cyan-300 hover:bg-cyan-950/40 hover:border-cyan-400 hover:shadow-[0_0_10px_rgba(0,240,255,0.3)] cursor-pointer"
              : "bg-black/40 border-neutral-900 text-neutral-600 cursor-not-allowed"
          }`}
          title="Undo last word (Ctrl+Z)"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>ወደ ኋላ (Undo)</span>
        </button>

        <button
          type="button"
          onClick={onReset}
          disabled={!hasContent}
          className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
            hasContent
              ? "bg-black border-neutral-800 text-rose-300 hover:border-rose-600 hover:bg-rose-950/30 cursor-pointer"
              : "bg-black/40 border-neutral-900 text-neutral-600 cursor-not-allowed"
          }`}
          title="Clear sentence (Esc)"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>አጽዳ (Reset)</span>
        </button>
      </div>

      <button
        type="button"
        onClick={handleCopy}
        disabled={!hasContent}
        className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold border-2 transition-all ${
          hasContent
            ? "bg-black border-amber-500/80 text-amber-300 hover:bg-amber-400 hover:text-black hover:shadow-[0_0_15px_rgba(255,215,0,0.4)] cursor-pointer"
            : "bg-black/40 border-neutral-900 text-neutral-600 cursor-not-allowed"
        }`}
      >
        {copied ? (
          <>
            <Check className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-300 font-bold">ተገልብጧል (Copied!)</span>
          </>
        ) : (
          <>
            <Copy className="w-4 h-4 text-amber-400" />
            <span>ቅዳ (Copy Sentence)</span>
          </>
        )}
      </button>
    </div>
  );
}
