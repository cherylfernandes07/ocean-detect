/**
 * WhaleJournal.tsx
 *
 * A running log of every quote generated in the current session,
 * grouped by orca name — the "whale journal" panel.
 *
 * WHY THIS EXISTS:
 *   Over a monitoring session you might detect multiple orcas or the same
 *   orca many times. This panel collects every generated quote so you can
 *   scroll back and read the full "conversation" from each whale across
 *   the session — like a field journal an Orcasound researcher would keep.
 *
 * DATA FLOW:
 *   Parent page maintains a `JournalEntry[]` array in state.
 *   Each time /workflow returns, a new entry is appended.
 *   This component just renders that list — no internal state needed.
 *
 * GROUPING:
 *   Entries are grouped by orca name in insertion order. Within each group,
 *   they're shown newest-first so the latest quote is always at the top.
 *
 * Accessibility:
 *   - aria-live="polite" so screen readers announce each new entry
 *   - role="feed" for a stream of articles that updates over time
 *   - Each article has aria-label with orca name + timestamp
 */

import React from "react";

export interface JournalEntry {
  id: string;           // unique key, e.g. Date.now().toString()
  orcaName: string;     // e.g. "Tahlequah"
  callType?: string;    // e.g. "S19"
  quote: string;        // the generated whale voice text
  timestamp: string;    // human-readable, e.g. "14:32:07"
  approved: boolean;    // did the LLM moderation step approve this post?
}

interface WhaleJournalProps {
  entries: JournalEntry[];
}

export default function WhaleJournal({ entries }: WhaleJournalProps) {
  if (entries.length === 0) {
    return (
      <section
        aria-label="Whale journal"
        className="rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-lg"
      >
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-slate-400">
          Whale Journal
        </h2>
        <p className="text-slate-500 italic text-sm">
          No entries yet. Quotes will appear here as orcas are detected.
        </p>
      </section>
    );
  }

  // Group entries by orca name, preserving the order each orca first appeared
  const grouped = entries.reduce<Record<string, JournalEntry[]>>((acc, entry) => {
    if (!acc[entry.orcaName]) acc[entry.orcaName] = [];
    acc[entry.orcaName].push(entry);
    return acc;
  }, {});

  // Get the unique orca names in the order they first appeared
  const orcaNames = Array.from(new Set(entries.map((e) => e.orcaName)));

  return (
    <section
      aria-label="Whale journal"
      className="rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-lg"
    >
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-400">
        Whale Journal
        {/* Entry count badge */}
        <span className="ml-2 rounded-full bg-slate-700 px-2 py-0.5 text-xs font-mono text-slate-300">
          {entries.length}
        </span>
      </h2>

      {/*
        role="feed" = a live region for a stream of articles.
        aria-live="polite" = announce new entries without interrupting.
      */}
      <div role="feed" aria-live="polite" aria-label="Orca quote entries" className="space-y-6">
        {orcaNames.map((orcaName) => {
          // Show newest entry first within each orca group
          const orcaEntries = [...grouped[orcaName]].reverse();

          return (
            <div key={orcaName}>
              {/* Orca name header with entry count */}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-base font-bold text-cyan-300">{orcaName}</span>
                <span className="text-xs text-slate-500">
                  {orcaEntries.length} {orcaEntries.length === 1 ? "entry" : "entries"}
                </span>
              </div>

              <div className="space-y-3 pl-3 border-l border-cyan-800/40">
                {orcaEntries.map((entry) => (
                  <article
                    key={entry.id}
                    aria-label={`${entry.orcaName} quote at ${entry.timestamp}`}
                    className="group"
                  >
                    {/* Quote text */}
                    <p className="text-sm text-slate-300 italic leading-relaxed">
                      "{entry.quote}"
                    </p>

                    {/* Metadata row */}
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                      <time dateTime={entry.timestamp}>{entry.timestamp}</time>

                      {entry.callType && (
                        <span className="font-mono text-cyan-600">{entry.callType}</span>
                      )}

                      {/* Moderation badge — did this get posted? */}
                      {entry.approved ? (
                        <span className="text-emerald-600">✓ posted</span>
                      ) : (
                        <span className="text-rose-700">✗ held</span>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
