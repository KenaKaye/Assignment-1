"use client";

import React from "react";
import { Sparkles } from "lucide-react";

interface QuickPromptsProps {
  onSelectPrompt: (prompt: string) => void;
}

const SAMPLE_PROMPTS = [
  "ኢትዮጵያ ውስጥ የቡና",
  "የኢትዮጵያ ፌዴራላዊ",
  "ጠቅላይ ሚኒስትር",
  "በአዲስ አበባ ከተማ",
  "የትምህርት እና የጤና",
];

export function QuickPrompts({ onSelectPrompt }: QuickPromptsProps) {
  return (
    <div className="space-y-2 pt-2">
      <div className="flex items-center gap-1.5 text-xs text-slate-400 font-semibold px-1">
        <Sparkles className="w-3.5 h-3.5 text-amber-400 drop-shadow-[0_0_5px_rgba(255,215,0,0.6)]" />
        <span className="text-amber-300/90 font-medium">የናሙና አጀማመሮች (Try Sample Prompts):</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {SAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onSelectPrompt(prompt)}
            className="px-3.5 py-2 rounded-xl bg-black hover:bg-neutral-950 text-slate-300 hover:text-cyan-300 border border-neutral-800 hover:border-cyan-400 text-xs font-bold transition-all duration-150 cursor-pointer shadow-sm hover:shadow-[0_0_12px_rgba(0,240,255,0.3)]"
            style={{ fontFamily: "var(--font-ethiopic), system-ui, sans-serif" }}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
