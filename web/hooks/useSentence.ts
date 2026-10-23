"use client";

import { useReducer, useCallback } from "react";

interface SentenceState {
  sentence: string;
  history: string[];
}

type SentenceAction =
  | { type: "SET"; payload: string }
  | { type: "APPEND"; payload: string }
  | { type: "UNDO" }
  | { type: "RESET" };

function sentenceReducer(state: SentenceState, action: SentenceAction): SentenceState {
  switch (action.type) {
    case "SET":
      return {
        sentence: action.payload,
        history: state.history,
      };
    case "APPEND": {
      const trimmed = state.sentence.trim();
      const nextSentence = trimmed ? `${trimmed} ${action.payload}` : action.payload;
      return {
        sentence: nextSentence,
        history: [...state.history, state.sentence],
      };
    }
    case "UNDO": {
      if (state.history.length === 0) return state;
      const previous = state.history[state.history.length - 1];
      return {
        sentence: previous,
        history: state.history.slice(0, -1),
      };
    }
    case "RESET":
      return {
        sentence: "",
        history: [],
      };
    default:
      return state;
  }
}

export function useSentence(initial: string = "") {
  const [state, dispatch] = useReducer(sentenceReducer, {
    sentence: initial,
    history: [],
  });

  const setSentence = useCallback((text: string) => {
    dispatch({ type: "SET", payload: text });
  }, []);

  const appendWord = useCallback((word: string) => {
    dispatch({ type: "APPEND", payload: word });
  }, []);

  const undo = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  return {
    sentence: state.sentence,
    canUndo: state.history.length > 0,
    setSentence,
    appendWord,
    undo,
    reset,
  };
}
