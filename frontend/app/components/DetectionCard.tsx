/**
 * DetectionCard.tsx
 *
 * Displays the result of a single orca detection:
 *   - Confidence score (as a percentage bar)
 *   - Detected / Not Detected status
 *   - Orca name and call type
 *
 * Accessibility:
 *   - aria-live="polite" so screen readers announce new detections
 *   - role="status" for the detection badge
 *   - Semantic <article> wrapper
 */

import React from "react";

export interface DetectionResult {
  detected: boolean;
  confidence: number;      // 0.0 – 1.0
  orca_name?: string;      // e.g. "Tahlequah"
  call_type?: string;      // e.g. "S_19"
  label?: string;          // "orca" | "not_orca"
  timestamp?: string;      // ISO string or human-readable
}

interface DetectionCardProps {
  result: DetectionResult | null;
  isLoading?: boolean;
}

export default function DetectionCard({ result, isLoading }: DetectionCardProps) {
  return (
    <article
      className="rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-lg"
      aria-label="Detection result"
    >
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-400">
        Detection
      </h2>

      {/* Screen readers will announce changes to this region */}
      <div aria-live="polite" aria-atomic="true">
        {isLoading && (
          <div className="flex items-center gap-3 text-slate-300">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
            <span>Analyzing audio…</span>
          </div>
        )}

        {!isLoading && !result && (
          <p className="text-slate-500 italic">No detection yet. Upload a WAV file to begin.</p>
        )}

        {!isLoading && result && (
          <div className="space-y-4">
            {/* Status badge */}
            <div
              role="status"
              className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-semibold ${
                result.detected
                  ? "bg-emerald-500/20 text-emerald-400"
                  : "bg-slate-600/40 text-slate-400"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  result.detected ? "bg-emerald-400 animate-pulse" : "bg-slate-500"
                }`}
              />
              {result.detected ? "Orca Detected" : "No Orca Detected"}
            </div>

            {/* Confidence bar */}
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-400">
                <span>Confidence</span>
                <span>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div
                className="h-2 w-full rounded-full bg-slate-700"
                role="progressbar"
                aria-valuenow={Math.round(result.confidence * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Detection confidence: ${(result.confidence * 100).toFixed(1)}%`}
              >
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    result.detected ? "bg-emerald-400" : "bg-slate-500"
                  }`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Orca details — only shown when detected */}
            {result.detected && (result.orca_name || result.call_type) && (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                {result.orca_name && (
                  <>
                    <dt className="text-slate-400">Orca</dt>
                    <dd className="font-medium text-white">{result.orca_name}</dd>
                  </>
                )}
                {result.call_type && (
                  <>
                    <dt className="text-slate-400">Call type</dt>
                    <dd className="font-medium text-white">{result.call_type}</dd>
                  </>
                )}
                {result.timestamp && (
                  <>
                    <dt className="text-slate-400">Time</dt>
                    <dd className="text-slate-300">
                      <time dateTime={result.timestamp}>{result.timestamp}</time>
                    </dd>
                  </>
                )}
              </dl>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
