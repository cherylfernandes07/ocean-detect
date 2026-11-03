/**
 * DetectionCard.stories.tsx
 *
 * Storybook stories for the DetectionCard component.
 *
 * WHAT IS A STORY?
 *   A story is just: "here is this component with these specific props."
 *   Storybook renders each story in isolation — no API calls, no state,
 *   no network. This lets you build and tweak UI without needing the backend.
 *
 *   Think of it like a photo shoot: you pose the component with mock data
 *   and capture each "look" separately. Later, designers and developers
 *   browse these like a lookbook.
 *
 * THE THREE STORIES HERE:
 *   1. Loading     — spinner while the CNN is running
 *   2. Detected    — orca found, high confidence, with name and call type
 *   3. NotDetected — no orca, low confidence
 *   4. Empty       — initial state, no result yet
 */

import type { Meta, StoryObj } from "@storybook/react";
import DetectionCard from "../app/components/DetectionCard";

// Meta = the "chapter heading" for this component in Storybook's sidebar
const meta: Meta<typeof DetectionCard> = {
  title: "OceanDetect/DetectionCard",   // sidebar path: OceanDetect > DetectionCard
  component: DetectionCard,
  parameters: {
    // Dark background to match the app's dark theme
    backgrounds: { default: "dark" },
    layout: "padded",
  },
  // argTypes lets Storybook generate controls in the UI for each prop
  argTypes: {
    isLoading: { control: "boolean" },
    result: { control: "object" },
  },
};

export default meta;
type Story = StoryObj<typeof DetectionCard>;

// ── Story 1: Loading state ────────────────────────────────────────────────────
export const Loading: Story = {
  args: {
    result: null,
    isLoading: true,
  },
};

// ── Story 2: Orca detected ────────────────────────────────────────────────────
export const Detected: Story = {
  args: {
    isLoading: false,
    result: {
      detected: true,
      confidence: 0.87,
      orca_name: "Tahlequah",
      call_type: "S19",
      label: "orca",
      timestamp: "14:32:07",
    },
  },
};

// ── Story 3: Not detected ─────────────────────────────────────────────────────
export const NotDetected: Story = {
  args: {
    isLoading: false,
    result: {
      detected: false,
      confidence: 0.12,
      label: "not_orca",
      timestamp: "14:31:02",
    },
  },
};

// ── Story 4: Empty (initial state before first upload) ────────────────────────
export const Empty: Story = {
  args: {
    result: null,
    isLoading: false,
  },
};

// ── Story 5: Borderline confidence ───────────────────────────────────────────
// Useful for testing the threshold boundary (0.3) visually
export const Borderline: Story = {
  name: "Borderline (near threshold)",
  args: {
    isLoading: false,
    result: {
      detected: true,
      confidence: 0.31,
      orca_name: "Mega",
      call_type: "N1",
      label: "orca",
      timestamp: "09:15:44",
    },
  },
};
