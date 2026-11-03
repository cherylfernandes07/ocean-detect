# 🐋 OceanDetect

> Real-time orca detection from live hydrophone audio, powered by a CNN + LangChain workflow that generates conservation-focused social media posts — automatically.

Built for the AWS Hackathon · MIT License · [Nirvaan Labs](https://github.com/NirvaanLabs)

---

## What it does

Southern Resident orcas are critically endangered. [Orcasound](https://www.orcasound.net) runs a network of underwater microphones (hydrophones) in Puget Sound that stream audio 24/7 — but someone has to be listening.

**OceanDetect** listens for you:

1. 🎧 **Detects** orca calls in live hydrophone audio using a trained CNN
2. 🐋 **Generates** an imagined orca "voice" using Claude (Anthropic) via LangChain
3. ✅ **Moderates** the post to ensure scientific integrity
4. 📘 **Posts** to Facebook automatically when a call is confirmed

The result: every orca detection becomes a conservation moment shared with the public — without a human having to be awake at 3am.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Live Audio Feed                          │
│              (hydrophone WAV · 20kHz · Puget Sound)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 1-second chunks
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window Vote                          │
│         simulate_live.py · 5-chunk window · 3/5 threshold       │
│         60-second cooldown after each detection fires           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ majority vote passed
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  POST /detect          OrcaCNN (PyTorch)                        │
│  ├─ load_audio_window  resample → normalize → pad to 2.45s      │
│  ├─ audio_to_specgram  Hanning FFT · 75% overlap · [71×1020]    │
│  └─ OrcaCNN forward    softmax → confidence → threshold 0.3     │
│                                                                 │
│  POST /workflow        LangChain RunnableSequence               │
│  ├─ run_whale_voice    Claude Haiku · temp 0.8 · orca persona   │
│  ├─ run_generate_post  Claude Haiku · conservation framing      │
│  ├─ run_moderate       Claude Haiku · temp 0.2 · fact-check     │
│  └─ run_post_to_facebook  Facebook Graph API · page feed        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ JSON response
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                             │
│                                                                 │
│  DetectionCard    confidence bar · detected badge               │
│  WhaleQuote       imagined orca voice · pull-quote card         │
│  PostPreview      draft post · moderation status · FB badge     │
│  WhaleJournal     session log · all quotes grouped by orca      │
│                                                                 │
│  useTransition    non-blocking CNN + LLM calls                  │
│  useDeferredValue deferred confidence bar updates               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Audio ML | PyTorch · OrcaCNN · scipy FFT |
| AI chain | LangChain · Claude Haiku (Anthropic) |
| Backend | FastAPI · Python 3.9+ |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS |
| Testing | Playwright E2E · Storybook component library |
| CI | GitHub Actions |
| Social | Facebook Graph API |
| Data | [Orcasound](https://www.orcasound.net) public hydrophone archive |

---

## Quick start

### Prerequisites

- Python 3.9+
- Node.js 20+
- An [Anthropic API key](https://console.anthropic.com)

### 1. Clone

```bash
git clone https://github.com/NirvaanLabs/ocean-detect.git
cd ocean-detect
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
MODEL_PATH=model_best_f1.pt

# Optional — needed to actually post to Facebook
FACEBOOK_PAGE_TOKEN=your_page_access_token
FACEBOOK_PAGE_ID=your_page_id
```

Start the API:

```bash
uvicorn backend.main:app --reload
```

The API runs at `http://localhost:8000`. Check `http://localhost:8000/health` to confirm.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## Running the live simulator

`simulate_live.py` replays a WAV file one second at a time, simulating a real hydrophone feed:

```bash
# Basic replay
python simulate_live.py --wav /path/to/recording.wav

# Loop forever + specify orca
python simulate_live.py --wav recording.wav --orca Tahlequah --call S19 --loop

# Bypass CNN (useful when model needs retraining)
python simulate_live.py --wav recording.wav --force
```

The terminal shows a live sliding window:
```
  [14:32:07] chunk    5 | conf=0.87 | votes=3/5 🟢🟢⚫🟢🟢
  [14:32:07] 🚨 Threshold met (3/5) — triggering workflow!
  ┌─────────────────────────────────────────────────────
  │ 🐋 ORCA DETECTED — confidence 87%
  │ 🎵 Whale voice:
  │   The water remembers every fin that has passed through it.
  │ ✅ Moderation: APPROVED
  └─────────────────────────────────────────────────────
  [14:32:37] ⏳ Cooldown started — next detection in 60s
```

---

## API reference

### `GET /health`
```json
{ "status": "ok", "device": "cpu", "model": "model_best_f1.pt" }
```

### `POST /detect`
Upload a WAV file. Returns CNN detection result.
```json
{ "detected": true, "confidence": 0.87, "label": "orca" }
```

### `POST /workflow`
Upload a WAV file. Runs the full LangChain chain.

Query params: `orca_name`, `call_type`, `location`, `force` (bypass CNN)

```json
{
  "detected": true,
  "confidence": 0.87,
  "whale_voice": "The water remembers every fin...",
  "draft_post": "🐋 Orca detected!...",
  "approved": true,
  "moderation_reason": "Post uses clear creative framing.",
  "final_post": "🐋 Orca detected at Orcasound Lab!...",
  "facebook_post_id": null,
  "facebook_error": "FACEBOOK_PAGE_TOKEN not set"
}
```

---

## Running tests

```bash
# Playwright E2E (starts Next.js automatically)
cd frontend
npx playwright test

# Visual test runner
npx playwright test --ui

# Storybook component library
npm run storybook
```

---

## Project structure

```
ocean-detect/
├── backend/
│   ├── main.py          # FastAPI app · /detect endpoint · spectrogram pipeline
│   ├── workflow.py      # LangChain chain · whale voice · moderation · Facebook
│   ├── whale_voice.py   # Standalone /whale-voice endpoint
│   └── model.py         # OrcaCNN architecture (PyTorch)
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # Three-panel layout · useTransition · useDeferredValue
│   │   └── components/
│   │       ├── DetectionCard.tsx # Confidence bar · detected badge
│   │       ├── WhaleQuote.tsx    # Orca voice pull-quote
│   │       ├── PostPreview.tsx   # Draft post · moderation status
│   │       └── WhaleJournal.tsx  # Session log grouped by orca
│   └── stories/                  # Storybook stories (one per component)
├── tests/e2e/           # Playwright E2E tests
├── simulate_live.py     # Live hydrophone simulator · sliding window vote
├── train.py             # CNN training script
└── .github/workflows/   # GitHub Actions CI
```

---

## The moderation step

A second LLM pass reviews every post before it's published. It rejects posts that:

- Present creative whale quotes as factual statements
- Attribute specific emotions as scientific fact
- Use language not supported by current cetacean research

This ensures OceanDetect maintains scientific integrity while still being emotionally engaging. The moderation model runs at `temperature=0.2` (conservative) while the creative steps run at `temperature=0.8`.

---

## Facebook setup

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Create an app → Add "Pages" product
3. Generate a Page Access Token for your conservation page
4. Add to `.env`:
   ```
   FACEBOOK_PAGE_TOKEN=your_token
   FACEBOOK_PAGE_ID=your_page_id
   ```

Posts will appear automatically on your page when a detection is confirmed and approved.

---

## Data sources

- **Training data**: [Orcasound public S3 archive](https://github.com/orcasound/orcadata) — labeled orca call clips from Southern Resident J, K, and L pods
- **Live stream**: Orcasound hydrophone network, Puget Sound, Washington
- **Orca profiles**: Based on [The Whale Museum](https://whalemuseum.org) and [Center for Whale Research](https://www.whaleresearch.com) identification records

---

## Contributing

This is an open-source project under the [Nirvaan Labs](https://github.com/NirvaanLabs) GitHub org. Contributions welcome:

- 🧠 Better CNN architecture or training data
- 🐋 More orca personality profiles
- 🌊 Support for additional hydrophone networks (MBARI, ONC)
- 📱 Twitter/Instagram posting support

---

## Cost efficiency

One of the core design goals of OceanDetect is to make 24/7 hydrophone monitoring economically viable. LLMs are powerful but expensive — they should be the **last resort**, not the first line of defense.

OceanDetect uses a layered filtering strategy:

| Layer | Tool | Cost | Purpose |
|---|---|---|---|
| 1st | CNN / decision model | Very low | Filters raw audio — runs on every 1-second chunk |
| 2nd | Sliding window vote | Free | Requires 3/5 chunks positive before escalating |
| 3rd | LLM (Claude Haiku) | Higher | Only called when a real detection is confirmed |

This means the LLM chain fires a fraction of the time it would without filtering — keeping costs low while preserving the creative and moderation quality only an LLM can provide.

### The math

```
savings = (1 - detection_rate) × cost_per_LLM_call × total_chunks
```

A 24/7 hydrophone produces 86,400 one-second chunks per day. Orca calls are present in roughly 1–5% of that audio in active monitoring zones.

| Scenario | LLM calls / day | Estimated cost |
|---|---|---|
| No detection model (every chunk) | 86,400 | ~$864 |
| With CNN (2% detection rate) | 1,728 | ~$17 |
| **Savings** | | **~$847 (98%)** |

*Based on Claude Haiku pricing (~$0.01 per workflow call). Actual costs vary.*

### How to measure your detection rate

Run the simulator on a long recording and check the terminal output:

```
detection_rate = workflow triggers fired / total chunks processed
```

The lower the detection rate, the greater the savings — and the more a detection model pays for itself. For a network of hydrophones monitoring multiple locations simultaneously, the cost difference becomes even more dramatic.

---

## Known issues

- **Model accuracy is low** — the CNN was trained on a limited dataset and has a high false-negative rate on real recordings. The `?force=true` flag exists to bypass it during testing. Retraining with more labeled data is the top priority.
- **`?force=true` in frontend** — the Run Detection button currently bypasses the CNN. Remove `&force=true` from the fetch URL in `app/page.tsx` once the model is stronger.
- **Facebook not yet live** — the posting step is fully wired but requires a real Page Access Token. Posts are approved by the moderation step but silently skipped until credentials are added to `.env`.
- **No streaming support** — `simulate_live.py` replays a pre-recorded WAV file. A real hydrophone stream (WebSocket or HLS) is not yet connected.
- **Python 3.9 only** — the backend venv was built on 3.9. Some newer LangChain features require 3.11+.

---

## Future work

- **Model retraining** — collect more labeled orca/non-orca clips from the Orcasound archive and retrain with data augmentation (time-stretch, pitch-shift, noise injection) to improve recall
- **Live stream connection** — connect directly to Orcasound's HLS stream instead of replaying WAV files; process audio in real time as it arrives
- **More orca profiles** — add personality prompts for other named Southern Residents (Granny, Mega, Notch, etc.) so the whale voice reflects the individual
- **Twitter / Instagram support** — extend `run_post_to_facebook` into a multi-platform posting step
- **MBARI hydrophone support** — add support for the Monterey Bay Accelerating Research Technology hydrophone network off the California coast
- **Hugging Face model publish** — upload `model_best_f1.pt` and the training script so the community can fine-tune or benchmark against it
- **Cloud deploy** — host the FastAPI backend on Render or Azure so the frontend can be used without running a local server
- **Multi-species support** — the spectrogram pipeline and LangChain chain are species-agnostic; extending to humpbacks or gray whales requires only new training data and personality prompts

---

## License

MIT · Built with ❤️ for the Southern Residents

*"The water remembers every fin that has passed through it."*