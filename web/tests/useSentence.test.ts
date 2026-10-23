import { renderHook, act } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { useSentence } from "@/hooks/useSentence";

describe("useSentence hook", () => {
  it("initializes with empty sentence and canUndo as false", () => {
    const { result } = renderHook(() => useSentence(""));
    expect(result.current.sentence).toBe("");
    expect(result.current.canUndo).toBe(false);
  });

  it("appends words properly with spacing", () => {
    const { result } = renderHook(() => useSentence("ኢትዮጵያ"));
    act(() => {
      result.current.appendWord("ውስጥ");
    });
    expect(result.current.sentence).toBe("ኢትዮጵያ ውስጥ");
    expect(result.current.canUndo).toBe(true);

    act(() => {
      result.current.appendWord("የቡና");
    });
    expect(result.current.sentence).toBe("ኢትዮጵያ ውስጥ የቡና");
  });

  it("supports undo to previous state", () => {
    const { result } = renderHook(() => useSentence("ኢትዮጵያ"));
    act(() => {
      result.current.appendWord("ውስጥ");
    });
    expect(result.current.sentence).toBe("ኢትዮጵያ ውስጥ");

    act(() => {
      result.current.undo();
    });
    expect(result.current.sentence).toBe("ኢትዮጵያ");
    expect(result.current.canUndo).toBe(false);
  });

  it("handles undo gracefully when history is empty", () => {
    const { result } = renderHook(() => useSentence(""));
    act(() => {
      result.current.undo();
    });
    expect(result.current.sentence).toBe("");
    expect(result.current.canUndo).toBe(false);
  });

  it("resets sentence and clears history", () => {
    const { result } = renderHook(() => useSentence("ኢትዮጵያ"));
    act(() => {
      result.current.appendWord("ውስጥ");
    });
    act(() => {
      result.current.reset();
    });
    expect(result.current.sentence).toBe("");
    expect(result.current.canUndo).toBe(false);
  });
});
