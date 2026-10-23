import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { SentenceEditor } from "@/components/SentenceEditor";
import { SuggestionList } from "@/components/SuggestionList";
import { FallbackNotice } from "@/components/FallbackNotice";
import { ErrorBanner } from "@/components/ErrorBanner";

describe("UI Components", () => {
  it("SentenceEditor renders and triggers onChange on typing", () => {
    const handleChange = vi.fn();
    render(
      <SentenceEditor
        value="ኢትዮጵያ"
        onChange={handleChange}
      />
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toBe("ኢትዮጵያ");

    fireEvent.change(textarea, { target: { value: "ኢትዮጵያ ውስጥ" } });
    expect(handleChange).toHaveBeenCalledWith("ኢትዮጵያ ውስጥ");
  });

  it("SuggestionList renders items and calls onSelect when clicked", () => {
    const handleSelect = vi.fn();
    const mockSuggestions = [
      { word: "መገኛ", probability: 0.40 },
      { word: "ቤት", probability: 0.30 },
      { word: "ላይ", probability: 0.15 },
    ];

    render(
      <SuggestionList
        suggestions={mockSuggestions}
        loading={false}
        onSelect={handleSelect}
        hasInput={true}
      />
    );

    expect(screen.getByText("መገኛ")).toBeInTheDocument();
    expect(screen.getByText("ቤት")).toBeInTheDocument();
    expect(screen.getByText("ላይ")).toBeInTheDocument();

    fireEvent.click(screen.getByText("መገኛ"));
    expect(handleSelect).toHaveBeenCalledWith("መገኛ");
  });

  it("FallbackNotice renders only when fallback is true", () => {
    const { rerender } = render(
      <FallbackNotice
        fallback={false}
        modelUsed={3}
        requestedN={3}
      />
    );
    expect(screen.queryByText(/የሞዴል ቅነሳ ተጠቅሟል/i)).toBeNull();

    rerender(
      <FallbackNotice
        fallback={true}
        modelUsed={2}
        requestedN={3}
        strategy="trigram_with_bigram_backoff"
      />
    );
    expect(screen.getByText(/የሞዴል ቅነሳ ተጠቅሟል/i)).toBeInTheDocument();
    expect(screen.getByText(/trigram_with_bigram_backoff/i)).toBeInTheDocument();
  });

  it("ErrorBanner displays error message when provided", () => {
    const { rerender } = render(<ErrorBanner message={null} />);
    expect(screen.queryByText(/የአገልግሎት ግንኙነት ችግር/i)).toBeNull();

    rerender(<ErrorBanner message="Backend connection failed." />);
    expect(screen.getByText(/የአገልግሎት ግንኙነት ችግር/i)).toBeInTheDocument();
    expect(screen.getByText("Backend connection failed.")).toBeInTheDocument();
  });
});
