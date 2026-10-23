"use client";

import React from "react";
import { AlertCircle, Terminal } from "lucide-react";

interface ErrorBannerProps {
  message: string | null;
}

export function ErrorBanner({ message }: ErrorBannerProps) {
  if (!message) return null;

  return (
    <div className="flex flex-col gap-2.5 p-4 rounded-2xl bg-black border border-rose-500/60 text-rose-200 text-xs shadow-[0_0_15px_rgba(244,63,94,0.25)]">
      <div className="flex items-center gap-2">
        <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
        <span className="font-bold text-rose-300">የአገልግሎት ግንኙነት ችግር (Service Connection Notice)</span>
      </div>
      <p className="pl-6 text-slate-300">{message}</p>
      <div className="ml-6 mt-1 flex items-center gap-2 p-2.5 rounded-xl bg-neutral-950 border border-cyan-500/30 text-cyan-300 font-mono text-[11px] shadow-inner">
        <Terminal className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
        <code>python -m uvicorn app.api:app --port 8000</code>
      </div>
    </div>
  );
}
