import { SuggestResponse, ApiError } from "./types";

export async function fetchSuggestions(
  text: string,
  n: number = 3,
  k: number = 5,
  signal?: AbortSignal
): Promise<SuggestResponse> {
  const params = new URLSearchParams({
    text: text.trim(),
    n: n.toString(),
    k: k.toString(),
  });

  const res = await fetch(`/api/suggest?${params.toString()}`, {
    signal,
    headers: {
      Accept: "application/json",
    },
  });

  if (!res.ok) {
    const errorBody: ApiError = await res.json().catch(() => ({
      error: "unknown_error",
      message: `Failed to fetch predictions (status ${res.status})`,
    }));
    throw new Error(errorBody.message || "Prediction error");
  }

  return (await res.json()) as SuggestResponse;
}
