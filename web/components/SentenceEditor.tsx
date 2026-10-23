"use client";

import React, { useRef, useEffect } from "react";

interface SentenceEditorProps {
  value: string;
  onChange: (value: string) => void;
  onSelectFirst?: () => void;
  onUndo?: () => void;
  onClear?: () => void;
}

export function SentenceEditor({
  value,
  onChange,
  onSelectFirst,
  onUndo,
  onClear,
}: SentenceEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl+Z or Cmd+Z -> Undo
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      onUndo?.();
      return;
    }

    // Escape -> Clear
    if (e.key === "Escape") {
      e.preventDefault();
      onClear?.();
      return;
    }

    // Tab -> accept top suggestion
    if (e.key === "Tab" && onSelectFirst) {
      e.preventDefault();
      onSelectFirst();
    }
  };

  const wordCount = value.trim() ? value.trim().split(/\s+/).length : 0;
  const charCount = value.length;

  return (
    <div className="relative w-full space-y-2">
      <div className="flex justify-between items-center px-1 text-xs">
        <label
          htmlFor="sentence-input"
          className="tracking-wider uppercase text-[11px] font-bold text-cyan-400 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]"
        >
          የአማርኛ ዓረፍተ ነገር (Amharic Sentence Editor)
        </label>
        <span className="tabular-nums font-mono text-[11px] text-amber-400/90 font-medium">
          {wordCount} {wordCount === 1 ? "word" : "words"} · {charCount} chars
        </span>
      </div>

      <div className="relative rounded-2xl bg-black border-2 border-neutral-800 shadow-2xl transition-all duration-200 focus-within:border-cyan-400 focus-within:ring-4 focus-within:ring-cyan-500/20">
        <textarea
          id="sentence-input"
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="እዚህ መተየብ ይጀምሩ... (Type or click sample prompts below)"
          rows={3}
          className="w-full resize-none bg-transparent px-5 py-4 text-xl leading-relaxed text-slate-100 placeholder:text-neutral-600 focus:outline-none"
          style={{ fontFamily: "var(--font-ethiopic), system-ui, sans-serif" }}
        />

        <div className="flex items-center justify-between px-4 py-2.5 bg-neutral-950/80 border-t border-neutral-900 rounded-b-2xl text-[11px]">
          <span className="flex items-center gap-1.5 text-slate-400">
            <kbd className="px-2 py-0.5 rounded bg-black text-cyan-300 font-mono text-[10px] border border-cyan-500/50 shadow-[0_0_5px_rgba(0,240,255,0.3)]">
              Tab
            </kbd>
            <span>accept top suggestion</span>
          </span>
          <span className="flex items-center gap-1.5 text-slate-400">
            <kbd className="px-2 py-0.5 rounded bg-black text-amber-300 font-mono text-[10px] border border-amber-500/50 shadow-[0_0_5px_rgba(255,215,0,0.3)]">
              1 - 5
            </kbd>
            <span>pick word</span>
          </span>
        </div>
      </div>
    </div>
  );
}
