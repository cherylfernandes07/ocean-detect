/**
 * detection.spec.ts — Playwright E2E tests for the OceanDetect frontend
 *
 * WHAT PLAYWRIGHT DOES:
 *   These tests launch a real browser, navigate to the app, and simulate
 *   exactly what a user would do — clicking, typing, uploading files.
 *   Unlike unit tests (which test one function), E2E tests verify the
 *   whole flow from click to API response to rendered result.
 *
 * MOCKING THE BACKEND:
 *   We don't want these tests to actually call the FastAPI backend — that
 *   would make them slow and fragile (backend might not be running in CI).
 *   Instead, we use Playwright's route() to intercept API calls and return
 *   fake responses. Think of it like a stunt double: the test sees a real
 *   API response, but we controlled exactly what it said.
 *
 * TESTS:
 *   1. Page loads with correct title
 *   2. File upload enables the Run button
 *   3. After detection: DetectionCard shows orca name
 *   4. After detection: WhaleQuote shows generated text
 *   5. After detection: WhaleJournal gains an entry
 *   6. Moderation rejected: PostPreview shows "Held" badge
 */

import { test, expect, Page } from "@playwright/test";
import path from "path";
import fs from "fs";

// ── Mock API responses ─────────────────────────────────────────────────────────

const MOCK_WORKFLOW_APPROVED = {
  detected: true,
  confidence: 0.87,
  whale_voice:
    "The water remembers every fin that has passed through it.",
  draft_post: "🐋 Orca detected!\n\n\"The water remembers every fin...\"\n\n#Orca",
  approved: true,
  moderation_reason:
    "Post clearly frames the orca quote as imaginative and includes conservation context.",
  final_post:
    "🐋 Orca detected at Orcasound Lab!\n\n" +
    "If we could imagine what Tahlequah might say:\n\n" +
    "\"The water remembers every fin that has passed through it.\"\n\n" +
    "Help protect Southern Resident orcas → orcasound.net\n#Orca",
  facebook_post_id: undefined,
  facebook_error: "FACEBOOK_PAGE_TOKEN not set",
};

const MOCK_WORKFLOW_REJECTED = {
  ...MOCK_WORKFLOW_APPROVED,
  approved: false,
  moderation_reason:
    "Post attributes specific emotions as established fact. Requires clearer creative framing.",
  final_post: MOCK_WORKFLOW_APPROVED.draft_post,
};

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Intercept POST /workflow and return a mock response.
 * Uses Playwright's route API — like a man-in-the-middle for HTTP calls.
 */
async function mockWorkflowRoute(page: Page, response = MOCK_WORKFLOW_APPROVED) {
  await page.route("**/workflow**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(response),
    });
  });
}

/**
 * Create a minimal WAV file in a temp path so we have something real to upload.
 * The content doesn't matter — the backend is mocked — but the file must exist.
 */
function createMinimalWavFile(): string {
  const tmpPath = path.join("/tmp", "test_orca.wav");
  // 44-byte WAV header for a 0-sample 16-bit mono 20kHz file
  // (enough for the browser's file picker to accept it as audio/wav)
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36, 4);        // chunk size
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);       // subchunk1 size
  header.writeUInt16LE(1, 20);        // PCM = 1
  header.writeUInt16LE(1, 22);        // 1 channel
  header.writeUInt32LE(20000, 24);    // sample rate
  header.writeUInt32LE(40000, 28);    // byte rate
  header.writeUInt16LE(2, 32);        // block align
  header.writeUInt16LE(16, 34);       // bits per sample
  header.write("data", 36);
  header.writeUInt32LE(0, 40);        // data size = 0 samples
  fs.writeFileSync(tmpPath, header);
  return tmpPath;
}

// ── Tests ──────────────────────────────────────────────────────────────────────

test.describe("OceanDetect frontend", () => {

  test.beforeEach(async ({ page }) => {
    await page.goto("http://localhost:3000");
  });

  // 1. Basic smoke test
  test("page loads with OceanDetect title", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "OceanDetect" })).toBeVisible();
    await expect(page.getByText("Orcasound Conservation")).toBeVisible();
  });

  // 2. File upload enables the Run button
  test("Run button is disabled until a file is selected", async ({ page }) => {
    const runButton = page.getByRole("button", { name: /Run Detection/i });

    // Initially disabled
    await expect(runButton).toBeDisabled();

    // Upload a WAV file
    const wavPath = createMinimalWavFile();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(wavPath);

    // Now enabled
    await expect(runButton).toBeEnabled();
  });

  // 3. Detection result appears in DetectionCard
  test("DetectionCard shows orca name after successful detection", async ({ page }) => {
    await mockWorkflowRoute(page);

    const wavPath = createMinimalWavFile();
    await page.locator('input[type="file"]').setInputFiles(wavPath);

    // Set orca name
    await page.fill('input[placeholder="e.g. Tahlequah"]', "Tahlequah");

    await page.getByRole("button", { name: /Run Detection/i }).click();

    // Wait for the detection result to appear
    await expect(page.getByText("Orca Detected")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("Tahlequah")).toBeVisible();
  });

  // 4. WhaleQuote appears with generated text
  test("WhaleQuote shows generated voice after detection", async ({ page }) => {
    await mockWorkflowRoute(page);

    const wavPath = createMinimalWavFile();
    await page.locator('input[type="file"]').setInputFiles(wavPath);
    await page.getByRole("button", { name: /Run Detection/i }).click();

    await expect(
      page.getByText("The water remembers every fin that has passed through it.")
    ).toBeVisible({ timeout: 10000 });
  });

  // 5. WhaleJournal gains an entry
  test("WhaleJournal shows entry after detection", async ({ page }) => {
    await mockWorkflowRoute(page);

    const wavPath = createMinimalWavFile();
    await page.locator('input[type="file"]').setInputFiles(wavPath);
    await page.getByRole("button", { name: /Run Detection/i }).click();

    // Journal should show the entry count badge "1"
    await expect(page.getByText("1")).toBeVisible({ timeout: 10000 });
    // And the orca name heading
    await expect(page.getByText("Tahlequah").first()).toBeVisible();
  });

  // 6. Moderation rejection shows "Held" badge
  test("PostPreview shows held badge when moderation rejects", async ({ page }) => {
    await mockWorkflowRoute(page, MOCK_WORKFLOW_REJECTED);

    const wavPath = createMinimalWavFile();
    await page.locator('input[type="file"]').setInputFiles(wavPath);
    await page.getByRole("button", { name: /Run Detection/i }).click();

    await expect(page.getByText("Held by moderation")).toBeVisible({ timeout: 10000 });
  });

});
