"use client";

import { useState, useEffect, useRef } from "react";
import { SuggestionItem } from "@/lib/types";
import { fetchSuggestions } from "@/lib/api";

interface UseSuggestionsResult {
  suggestions: SuggestionItem[];
  loading: boolean;
  error: string | null;
  modelUsed: number;
  fallback: boolean;
  strategy: string;
  latencyMs: number;
}

export function useSuggestions(sentence: string, n: number = 3): UseSuggestionsResult {
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [modelUsed, setModelUsed] = useState<number>(n);
  const [fallback, setFallback] = useState<boolean>(false);
  const [strategy, setStrategy] = useState<string>("none");
  const [latencyMs, setLatencyMs] = useState<number>(0);

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const trimmed = sentence.trim();

    // If input is empty, clear suggestions immediately
    if (!trimmed) {
      setSuggestions([]);
      setLoading(false);
      setError(null);
      setFallback(false);
      setStrategy("none");
      setLatencyMs(0);
      return;
    }

    // Cancel prior in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);

    // Debounce by 200 ms
    const timer = setTimeout(async () => {
      try {
        const res = await fetchSuggestions(trimmed, n, 5, controller.signal);
        setSuggestions(res.suggestions || []);
        setModelUsed(res.model_used);
        setFallback(res.fallback);
        setStrategy(res.strategy || "none");
        setLatencyMs(res.latency_ms || 0);
        setError(null);
      } catch (err: unknown) {
        if ((err as { name?: string })?.name === "AbortError") {
          return; // Ignore aborted requests
        }
        setError((err as Error).message || "Unable to fetch suggestions");
        setSuggestions([]);
      } finally {
        setLoading(false);
      }
    }, 200);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [sentence, n]);

  return {
    suggestions,
    loading,
    error,
    modelUsed,
    fallback,
    strategy,
    latencyMs,
  };
}
