/**
 * page.tsx — Main OceanDetect page
 *
 * THREE-PANEL LAYOUT:
 *   Left  — Upload WAV + DetectionCard (the CNN result)
 *   Center — WhaleQuote + PostPreview (the LLM chain result)
 *   Right  — WhaleJournal (running log of all quotes this session)
 *
 * REACT 18 CONCURRENT FEATURES USED HERE:
 *
 *   useTransition:
 *     Wraps the "Run Detection" click handler. This tells React that the
 *     state updates triggered by that button are "non-urgent" — React can
 *     interrupt them to handle more urgent things (like typing). Visually,
 *     it gives us isPending to show a spinner without blocking the UI.
 *     Think of it like telling a waiter "I'll have the salmon" — the order
 *     is placed (transition started), but you can still sip water while
 *     the kitchen works.
 *
 *   useDeferredValue:
 *     Used on the confidence value driving the spectrogram visualization.
 *     If the backend streams rapid updates, React can "lag behind" the
 *     spectrogram rendering without blocking the rest of the UI.
 *     Like a news ticker that's allowed to fall a sentence behind the
 *     live feed — the story is still accurate, just slightly delayed.
 */

"use client";

import React, { useState, useTransition, useDeferredValue, useCallback } from "react";
import DetectionCard, { DetectionResult } from "./components/DetectionCard";
import WhaleQuote from "./components/WhaleQuote";
import WhaleJournal, { JournalEntry } from "./components/WhaleJournal";
import PostPreview from "./components/PostPreview";

// Shape of the /workflow API response
interface WorkflowResult {
  detected: boolean;
  confidence: number;
  whale_voice: string;
  draft_post: string;
  approved: boolean;
  moderation_reason: string;
  final_post: string;
  facebook_post_id?: string;
  facebook_error?: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  // ── State ──────────────────────────────────────────────────────────────────

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [orcaName, setOrcaName]         = useState("Tahlequah");
  const [callType, setCallType]         = useState("S19");

  // Detection result from /detect or /workflow
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);

  // Latest workflow result (whale voice + post)
  const [workflowResult, setWorkflowResult] = useState<WorkflowResult | null>(null);

  // All quotes generated this session — the whale journal
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);

  // Error message to show if the API call fails
  const [error, setError] = useState<string | null>(null);

  // ── useTransition ──────────────────────────────────────────────────────────
  // isPending = true while the CNN + LLM chain is running
  // startTransition = wraps state updates that can be interrupted
  const [isPending, startTransition] = useTransition();

  // ── useDeferredValue ───────────────────────────────────────────────────────
  // Defers re-rendering the confidence bar slightly so rapid updates from
  // a streaming backend don't choke the main thread.
  const deferredConfidence = useDeferredValue(detectionResult?.confidence ?? 0);

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSelectedFile(e.target.files?.[0] ?? null);
      // Clear previous results when a new file is picked
      setDetectionResult(null);
      setWorkflowResult(null);
      setError(null);
    },
    []
  );

  /**
   * handleRunDetection
   *
   * Posts the selected WAV file to /workflow (which runs both CNN detection
   * and the full LangChain chain in one shot). On success, updates all three
   * panels. Wrapped in startTransition so the UI stays responsive.
   */
  const handleRunDetection = useCallback(() => {
    if (!selectedFile) return;

    setError(null);

    // startTransition marks this whole block as "non-urgent"
    // React can pause and resume it if something more urgent arrives
    startTransition(async () => {
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res = await fetch(
          `${API_BASE}/workflow?orca_name=${encodeURIComponent(orcaName)}&call_type=${encodeURIComponent(callType)}&force=true`,
          { method: "POST", body: formData }
        );

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`API error ${res.status}: ${text}`);
        }

        const data: WorkflowResult = await res.json();

        // Update detection panel
        setDetectionResult({
          detected: data.detected,
          confidence: data.confidence,
          orca_name: orcaName,
          call_type: callType,
          timestamp: new Date().toLocaleTimeString(),
        });

        // Update workflow panels
        setWorkflowResult(data);

        // Append to whale journal
        if (data.whale_voice) {
          const entry: JournalEntry = {
            id: Date.now().toString(),
            orcaName,
            callType,
            quote: data.whale_voice,
            timestamp: new Date().toLocaleTimeString(),
            approved: data.approved,
          };
          setJournalEntries((prev) => [...prev, entry]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    });
  }, [selectedFile, orcaName, callType]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Top nav */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl" aria-hidden="true">🐋</span>
        <h1 className="text-lg font-bold tracking-tight text-white">OceanDetect</h1>
        <span className="ml-auto text-xs text-slate-600">Orcasound Conservation</span>
      </header>

      {/* Three-panel grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6 max-w-7xl mx-auto">

        {/* ── LEFT PANEL: Upload + Detection ───────────────────────────────── */}
        <div className="space-y-4">
          {/* Upload card */}
          <section
            className="rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-lg"
            aria-label="Upload audio"
          >
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-400">
              Audio Upload
            </h2>

            {/* File picker */}
            <label
              className="block cursor-pointer rounded-xl border-2 border-dashed border-slate-600 p-6 text-center text-sm text-slate-400 hover:border-cyan-600 hover:text-cyan-400 transition-colors"
              htmlFor="wav-upload"
            >
              {selectedFile ? (
                <span className="text-cyan-300 font-medium">{selectedFile.name}</span>
              ) : (
                <>
                  <span className="block text-2xl mb-2">🎵</span>
                  Drop a WAV file or click to browse
                </>
              )}
            </label>
            <input
              id="wav-upload"
              type="file"
              accept=".wav,audio/wav"
              className="sr-only"
              onChange={handleFileChange}
              aria-label="Upload WAV file"
            />

            {/* Orca metadata inputs */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="orca-name" className="block text-xs text-slate-400 mb-1">
                  Orca name
                </label>
                <input
                  id="orca-name"
                  type="text"
                  value={orcaName}
                  onChange={(e) => setOrcaName(e.target.value)}
                  className="w-full rounded-lg bg-slate-900 border border-slate-600 px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                  placeholder="e.g. Tahlequah"
                />
              </div>
              <div>
                <label htmlFor="call-type" className="block text-xs text-slate-400 mb-1">
                  Call type
                </label>
                <input
                  id="call-type"
                  type="text"
                  value={callType}
                  onChange={(e) => setCallType(e.target.value)}
                  className="w-full rounded-lg bg-slate-900 border border-slate-600 px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                  placeholder="e.g. S19"
                />
              </div>
            </div>

            {/* Run button */}
            <button
              onClick={handleRunDetection}
              disabled={!selectedFile || isPending}
              className="mt-4 w-full rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed px-4 py-3 text-sm font-semibold text-white transition-colors flex items-center justify-center gap-2"
              aria-busy={isPending}
            >
              {isPending ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Running chain…
                </>
              ) : (
                "▶ Run Detection"
              )}
            </button>

            {/* Error */}
            {error && (
              <p role="alert" className="mt-3 text-xs text-rose-400 bg-rose-900/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}
          </section>

          {/* Detection result — passes deferredConfidence so rapid updates don't block */}
          <DetectionCard
            result={
              detectionResult
                ? { ...detectionResult, confidence: deferredConfidence }
                : null
            }
            isLoading={isPending}
          />
        </div>

        {/* ── CENTER PANEL: Whale Voice + Post Preview ──────────────────────── */}
        <div className="space-y-4">
          {workflowResult?.whale_voice ? (
            <WhaleQuote
              quote={workflowResult.whale_voice}
              orcaName={orcaName}
              callType={callType}
              timestamp={detectionResult?.timestamp}
              isGenerating={isPending}
            />
          ) : (
            <WhaleQuote
              quote=""
              orcaName={orcaName}
              isGenerating={isPending}
            />
          )}

          <PostPreview
            finalPost={workflowResult?.final_post ?? ""}
            approved={workflowResult?.approved ?? false}
            moderationReason={workflowResult?.moderation_reason}
            facebookPostId={workflowResult?.facebook_post_id}
            facebookError={workflowResult?.facebook_error}
            isLoading={isPending}
          />
        </div>

        {/* ── RIGHT PANEL: Whale Journal ────────────────────────────────────── */}
        <div>
          <WhaleJournal entries={journalEntries} />
        </div>
      </div>
    </main>
  );
}
