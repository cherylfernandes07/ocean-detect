/**
 * PostPreview.tsx
 *
 * Shows the draft Facebook post text before (and after) it's sent.
 * Includes moderation status badge and Facebook post ID if it was posted.
 *
 * WHY A PREVIEW COMPONENT:
 *   The /workflow chain generates a draft, moderates it, then (optionally)
 *   posts it. Users want to see what went out — or what was blocked and why.
 *   This component surfaces all three outcomes in one card.
 *
 * THREE STATES:
 *   1. approved=true, facebookPostId set  → green "Posted" badge + post ID
 *   2. approved=true, no facebookPostId   → green "Approved" badge (no FB token yet)
 *   3. approved=false                      → red "Held" badge + moderation reason
 *
 * Accessibility:
 *   - role="status" on the moderation badge so screen readers announce changes
 *   - <output> element for the post text (it's generated content, not user input)
 */

import React from "react";

export interface PostPreviewProps {
  /** The final post text (after moderation pass) */
  finalPost: string;
  /** Whether the moderation LLM approved this post */
  approved: boolean;
  /** Why moderation approved or rejected it */
  moderationReason?: string;
  /** Facebook post ID if successfully published, e.g. "123456_789012" */
  facebookPostId?: string;
  /** Error from Facebook API if posting failed */
  facebookError?: string;
  /** Whether the workflow is currently running */
  isLoading?: boolean;
}

export default function PostPreview({
  finalPost,
  approved,
  moderationReason,
  facebookPostId,
  facebookError,
  isLoading = false,
}: PostPreviewProps) {
  return (
    <article
      className="rounded-2xl border border-slate-700 bg-slate-800 p-6 shadow-lg"
      aria-label="Post preview"
    >
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-400">
        Post Preview
      </h2>

      {isLoading && (
        <div className="space-y-2 mb-4" aria-label="Generating post…">
          <div className="h-4 w-full animate-pulse rounded bg-slate-700" />
          <div className="h-4 w-5/6 animate-pulse rounded bg-slate-700" />
          <div className="h-4 w-3/4 animate-pulse rounded bg-slate-700" />
        </div>
      )}

      {!isLoading && (
        <>
          {/* Post text — <output> signals this is machine-generated content */}
          <output
            className="block mb-4 rounded-xl bg-slate-900/60 border border-slate-700/50 p-4 text-sm leading-relaxed text-slate-200 whitespace-pre-wrap"
            aria-label="Draft post text"
          >
            {finalPost || <span className="text-slate-500 italic">No post generated yet.</span>}
          </output>

          {/* Moderation status badge */}
          <div className="flex flex-wrap items-start gap-3">
            <div
              role="status"
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                approved
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-700/40"
                  : "bg-rose-500/20 text-rose-400 border border-rose-700/40"
              }`}
            >
              <span>{approved ? "✓" : "✗"}</span>
              {approved ? "Approved" : "Held by moderation"}
            </div>

            {/* Facebook outcome */}
            {approved && facebookPostId && (
              <div className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-700/40 px-3 py-1 text-xs font-semibold">
                📘 Posted · {facebookPostId}
              </div>
            )}

            {approved && !facebookPostId && facebookError && (
              <div
                className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-700/40 px-3 py-1 text-xs font-semibold"
                title={facebookError}
              >
                📘 Not posted
              </div>
            )}
          </div>

          {/* Moderation reason — shown in both approved and rejected cases */}
          {moderationReason && (
            <p className="mt-3 text-xs text-slate-500 italic">
              {moderationReason}
            </p>
          )}
        </>
      )}
    </article>
  );
}
