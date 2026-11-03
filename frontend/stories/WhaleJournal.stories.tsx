/**
 * WhaleJournal.stories.tsx
 *
 * Storybook stories for the WhaleJournal component.
 *
 * Stories cover:
 *   - Empty (no entries yet)
 *   - Single orca, multiple quotes
 *   - Multiple orcas (grouped)
 *   - Mix of approved and held posts
 */

import type { Meta, StoryObj } from "@storybook/react";
import WhaleJournal, { JournalEntry } from "../app/components/WhaleJournal";

const meta: Meta<typeof WhaleJournal> = {
  title: "OceanDetect/WhaleJournal",
  component: WhaleJournal,
  parameters: {
    backgrounds: { default: "dark" },
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof WhaleJournal>;

// ── Shared mock data ──────────────────────────────────────────────────────────

const tahlequahEntries: JournalEntry[] = [
  {
    id: "1",
    orcaName: "Tahlequah",
    callType: "S19",
    quote: "The water remembers every fin that has passed through it.",
    timestamp: "14:32:07",
    approved: true,
  },
  {
    id: "2",
    orcaName: "Tahlequah",
    callType: "S19",
    quote: "I call out and wait. Sometimes the echo returns.",
    timestamp: "14:35:22",
    approved: true,
  },
];

const mixedEntries: JournalEntry[] = [
  ...tahlequahEntries,
  {
    id: "3",
    orcaName: "Mega",
    callType: "N1",
    quote: "We are still here.",
    timestamp: "15:10:01",
    approved: false,  // this one was held by moderation
  },
  {
    id: "4",
    orcaName: "Granny",
    quote: "Every current carries memory.",
    timestamp: "15:12:45",
    approved: true,
  },
  {
    id: "5",
    orcaName: "Tahlequah",
    callType: "S4",
    quote: "The salmon run thin this year. We adapt.",
    timestamp: "15:30:00",
    approved: true,
  },
];

// ── Stories ───────────────────────────────────────────────────────────────────

export const Empty: Story = {
  args: { entries: [] },
};

export const SingleOrca: Story = {
  name: "Single orca (Tahlequah)",
  args: { entries: tahlequahEntries },
};

export const MultipleOrcas: Story = {
  name: "Multiple orcas + held post",
  args: { entries: mixedEntries },
};
