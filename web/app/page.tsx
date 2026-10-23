"use client";

import React, { useState, useEffect } from "react";
import { SentenceEditor } from "@/components/SentenceEditor";
import { SuggestionList } from "@/components/SuggestionList";
import { ModelControls } from "@/components/ModelControls";
import { ActionBar } from "@/components/ActionBar";
import { FallbackNotice } from "@/components/FallbackNotice";
import { ErrorBanner } from "@/components/ErrorBanner";
import { QuickPrompts } from "@/components/QuickPrompts";
import { useSentence } from "@/hooks/useSentence";
import { useSuggestions } from "@/hooks/useSuggestions";
import { BookOpen, Activity, Sparkles } from "lucide-react";

export default function Home() {
  const [n, setN] = useState<number>(3);
  const { sentence, canUndo, setSentence, appendWord, undo, reset } = useSentence("");
  const {
    suggestions,
    loading,
    error,
    modelUsed,
    fallback,
    strategy,
    latencyMs,
  } = useSuggestions(sentence, n);

  const [backendReady, setBackendReady] = useState<boolean | null>(null);

  // Health check on initial mount
  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/suggest?text=ኢትዮጵያ&n=2&k=1");
        if (mounted) {
          setBackendReady(res.ok);
        }
      } catch {
        if (mounted) {
          setBackendReady(false);
        }
      }
    };
    checkHealth();
    return () => {
      mounted = false;
    };
  }, []);

  const handleSelectFirst = () => {
    if (suggestions.length > 0) {
      appendWord(suggestions[0].word);
    }
  };

  return (
    <main className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 flex flex-col items-center justify-between bg-black text-slate-100">
      <div className="w-full max-w-2xl mx-auto space-y-7">
        {/* Header section */}
        <header className="space-y-3 text-center sm:text-left">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/80 border border-cyan-500/40 text-cyan-300 text-xs font-semibold tracking-wide shadow-[0_0_15px_rgba(0,240,255,0.25)]">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-spin" style={{ animationDuration: "6s" }} />
              <span>AAU SIT&E · NLP Assignment 1 (Q4)</span>
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span className="text-amber-400">N-gram Engine</span>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-neutral-950 border border-neutral-800 text-slate-400">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>API:</span>
                {backendReady === null ? (
                  <span className="text-slate-400">Checking...</span>
                ) : backendReady ? (
                  <span className="text-cyan-400 font-mono font-medium drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]">
                    Online (FastAPI)
                  </span>
                ) : (
                  <span className="text-amber-400 font-mono font-medium">
                    Offline
                  </span>
                )}
              </span>
            </div>
          </div>

          <div className="space-y-1.5">
            <h1
              className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white drop-shadow-[0_0_20px_rgba(0,240,255,0.35)]"
              style={{ fontFamily: "var(--font-ethiopic), system-ui, sans-serif" }}
            >
              የአማርኛ ቀጣይ ቃል <span className="text-cyan-400 text-glow-cyan">ጠቋሚ ስርዓት</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-400">
              Amharic Predictive Text & Next-Word Suggestion Engine · Trained on <span className="text-amber-400 font-semibold">GPAC Corpus</span>
            </p>
          </div>
        </header>

        {/* Main Application Card with Neon Border and Glassmorphism */}
        <div className="rounded-3xl bg-neutral-950/90 border-2 border-cyan-500/40 p-6 sm:p-8 glow-cyan-card backdrop-blur-xl space-y-6 relative overflow-hidden">
          {/* Subtle gold accent corner flare */}
          <div className="absolute -top-16 -right-16 w-36 h-36 bg-amber-400/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-16 -left-16 w-36 h-36 bg-cyan-400/10 rounded-full blur-3xl pointer-events-none" />

          {/* Model toggle controls */}
          <ModelControls
            n={n}
            onChange={setN}
            modelUsed={modelUsed}
            strategy={strategy}
            latencyMs={latencyMs}
          />

          {/* Connection Error Banner (if API is unreachable) */}
          <ErrorBanner message={error} />

          {/* Sentence Input Area */}
          <SentenceEditor
            value={sentence}
            onChange={setSentence}
            onSelectFirst={handleSelectFirst}
            onUndo={undo}
            onClear={reset}
          />

          {/* Fallback notification when backoff occurs */}
          <FallbackNotice
            fallback={fallback}
            strategy={strategy}
            modelUsed={modelUsed}
            requestedN={n}
          />

          {/* Suggestions List */}
          <SuggestionList
            suggestions={suggestions}
            loading={loading}
            onSelect={appendWord}
            hasInput={sentence.trim().length > 0}
          />

          {/* Action Bar (Undo, Reset, Copy) */}
          <ActionBar
            canUndo={canUndo}
            onUndo={undo}
            onReset={reset}
            sentence={sentence}
          />

          {/* Quick Prompts */}
          <QuickPrompts onSelectPrompt={setSentence} />
        </div>

        {/* Info card footer */}
        <footer className="rounded-2xl bg-black/60 border border-neutral-800/80 p-4 text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-inner">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>
              Trained on <span className="text-slate-200 font-semibold">90,000 sentences</span> from GPAC Amharic Corpus (<span className="text-amber-400 font-mono">188,263</span> vocabulary words).
            </span>
          </div>
          <div className="text-[11px] font-mono text-cyan-400/90 border border-cyan-500/30 px-2.5 py-0.5 rounded-md bg-cyan-950/30">
            Katz Backoff + Add-k Smoothing
          </div>
        </footer>
      </div>
    </main>
  );
}
