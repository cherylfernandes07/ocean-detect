/**
 * WhaleQuote.stories.tsx
 *
 * Storybook stories for the WhaleQuote component.
 *
 * Stories cover:
 *   - Generating (loading shimmer while LLM streams)
 *   - Short quote
 *   - Long quote (tests wrapping)
 *   - Quote without call type or timestamp
 */

import type { Meta, StoryObj } from "@storybook/react";
import WhaleQuote from "../app/components/WhaleQuote";

const meta: Meta<typeof WhaleQuote> = {
  title: "OceanDetect/WhaleQuote",
  component: WhaleQuote,
  parameters: {
    backgrounds: { default: "dark" },
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof WhaleQuote>;

// ── Story 1: Generating (LLM is streaming) ────────────────────────────────────
export const Generating: Story = {
  args: {
    quote: "",
    orcaName: "Tahlequah",
    callType: "S19",
    isGenerating: true,
  },
};

// ── Story 2: Short quote ──────────────────────────────────────────────────────
export const ShortQuote: Story = {
  args: {
    quote: "The water remembers every fin that has passed through it.",
    orcaName: "Tahlequah",
    callType: "S19",
    timestamp: "14:32:07",
    isGenerating: false,
  },
};

// ── Story 3: Long quote (tests text wrapping) ─────────────────────────────────
export const LongQuote: Story = {
  args: {
    quote:
      "If we could imagine what I might say in this moment — I am here, " +
      "in the waters my family has always known. Each click and whistle " +
      "carries the names of those who swam before me. The salmon are fewer " +
      "now, but the memory of them is deep, written in the currents " +
      "themselves. I call out and wait. Sometimes the echo returns.",
    orcaName: "Mega",
    callType: "N1",
    timestamp: "09:15:44",
    isGenerating: false,
  },
};

// ── Story 4: Minimal (no call type or timestamp) ──────────────────────────────
export const Minimal: Story = {
  args: {
    quote: "We are still here.",
    orcaName: "Granny",
    isGenerating: false,
  },
};
