/**
 * Shared types for Amharic Predictive Text UI
 */

export interface SuggestionItem {
  word: string;
  probability: number;
}

export interface SuggestResponse {
  input_tail: string[];
  requested_n: number;
  model_used: number;
  fallback: boolean;
  strategy?: string;
  latency_ms?: number;
  suggestions: SuggestionItem[];
}

export interface ApiError {
  error: string;
  message: string;
}
