"use client";

import React, { useEffect } from "react";
import { SuggestionItem } from "@/lib/types";
import { SuggestionChip } from "./SuggestionChip";

interface SuggestionListProps {
  suggestions: SuggestionItem[];
  loading: boolean;
  onSelect: (word: string) => void;
  hasInput: boolean;
}

export function SuggestionList({
  suggestions,
  loading,
  onSelect,
  hasInput,
}: SuggestionListProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeTag = document.activeElement?.tagName.toLowerCase();
      if (activeTag === "textarea" || activeTag === "input") {
        return;
      }

      const keyNum = parseInt(e.key, 10);
      if (!isNaN(keyNum) && keyNum >= 1 && keyNum <= suggestions.length) {
        e.preventDefault();
        onSelect(suggestions[keyNum - 1].word);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [suggestions, onSelect]);

  return (
    <div className="w-full space-y-2.5" aria-live="polite">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs uppercase tracking-wider font-bold text-cyan-400 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]">
          የሚቀጥሉ ቃላት ጥቆማ (Next-Word Predictions)
        </h3>
        {loading && (
          <span className="flex items-center gap-1.5 text-xs text-amber-400 animate-pulse font-mono font-medium">
            <span className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(255,215,0,0.8)]" />
            Computing...
          </span>
        )}
      </div>

      {loading && suggestions.length === 0 ? (
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-14 w-full rounded-2xl bg-neutral-950 border border-cyan-500/20 animate-pulse"
            />
          ))}
        </div>
      ) : suggestions.length > 0 ? (
        <div className="space-y-2.5">
          {suggestions.map((item, index) => (
            <SuggestionChip
              key={`${item.word}-${index}`}
              index={index}
              item={item}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : hasInput ? (
        <div className="rounded-2xl border border-dashed border-amber-500/30 bg-black/60 p-6 text-center text-slate-400 text-sm">
          ምንም ተዛማጅ ቃል አልተገኘም (No suggestions found for this context).
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-cyan-500/30 bg-black/60 p-8 text-center text-slate-400 text-sm">
          የሚቀጥለውን ቃል ጥቆማ ለማየት ዓረፍተ ነገር መተየብ ይጀምሩ።
          <div className="text-xs text-cyan-400/80 mt-1.5">
            Type an Amharic sentence or click a sample prompt below to see predictions.
          </div>
        </div>
      )}
    </div>
  );
}
