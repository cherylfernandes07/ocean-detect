/**
 * PostPreview.stories.tsx
 *
 * Storybook stories for the PostPreview component.
 *
 * Stories cover:
 *   - Loading (workflow running)
 *   - Approved + posted to Facebook
 *   - Approved but no Facebook token
 *   - Rejected by moderation
 */

import type { Meta, StoryObj } from "@storybook/react";
import PostPreview from "../app/components/PostPreview";

const meta: Meta<typeof PostPreview> = {
  title: "OceanDetect/PostPreview",
  component: PostPreview,
  parameters: {
    backgrounds: { default: "dark" },
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof PostPreview>;

// Shared post text for approved stories
const SAMPLE_POST =
  `🐋 Orca detected at Orcasound Lab hydrophone, Puget Sound!\n\n` +
  `If we could imagine what Tahlequah might say in this moment:\n\n` +
  `"The water remembers every fin that has passed through it. ` +
  `I call out and wait. Sometimes the echo returns."\n\n` +
  `Help protect Southern Resident orcas → orcasound.net\n` +
  `#Orca #Orcasound #MarineMammals #PugetSound`;

// ── Story 1: Loading ──────────────────────────────────────────────────────────
export const Loading: Story = {
  args: {
    finalPost: "",
    approved: false,
    isLoading: true,
  },
};

// ── Story 2: Approved + posted to Facebook ────────────────────────────────────
export const ApprovedAndPosted: Story = {
  name: "Approved — posted to Facebook",
  args: {
    finalPost: SAMPLE_POST,
    approved: true,
    moderationReason:
      "Post clearly frames the orca quote as imaginative and includes conservation context.",
    facebookPostId: "123456789_987654321",
    isLoading: false,
  },
};

// ── Story 3: Approved but no Facebook token ───────────────────────────────────
export const ApprovedNotPosted: Story = {
  name: "Approved — no Facebook token",
  args: {
    finalPost: SAMPLE_POST,
    approved: true,
    moderationReason: "Post is appropriate for publication.",
    facebookError: "FACEBOOK_PAGE_TOKEN not set in environment",
    isLoading: false,
  },
};

// ── Story 4: Rejected by moderation ──────────────────────────────────────────
export const Rejected: Story = {
  name: "Held by moderation",
  args: {
    finalPost:
      `Tahlequah is grieving. In her own words: "My calf is gone and ` +
      `I am devastated. The waters are empty without her."`,
    approved: false,
    moderationReason:
      "Post attributes specific emotional states as established facts. " +
      "Anthropomorphizing grief as fact rather than creative framing is not supported by science.",
    isLoading: false,
  },
};
