import { NextRequest, NextResponse } from "next/server";
import { SuggestResponse, ApiError } from "@/lib/types";

const BACKEND_URL = process.env.BACKEND_API_URL || "http://127.0.0.1:8000";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const text = searchParams.get("text");
  const n = searchParams.get("n") || "3";
  const k = searchParams.get("k") || "5";

  if (!text || text.trim().length === 0) {
    return NextResponse.json<ApiError>(
      { error: "invalid_input", message: "Text parameter is required." },
      { status: 400 }
    );
  }

  const targetUrl = `${BACKEND_URL}/suggest?text=${encodeURIComponent(text.trim())}&n=${n}&k=${k}`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(targetUrl, {
      signal: controller.signal,
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      return NextResponse.json<ApiError>(
        {
          error: errorData?.detail?.error || "backend_error",
          message: errorData?.detail?.message || `Backend returned status ${response.status}`,
        },
        { status: response.status }
      );
    }

    const data: SuggestResponse = await response.json();
    return NextResponse.json<SuggestResponse>(data);
  } catch (err: unknown) {
    const isAbort = (err as { name?: string })?.name === "AbortError";
    return NextResponse.json<ApiError>(
      {
        error: "service_unavailable",
        message: isAbort
          ? "Prediction backend timed out."
          : "The prediction service is not reachable. Start it with: python -m uvicorn app.api:app --port 8000",
      },
      { status: 503 }
    );
  }
}
