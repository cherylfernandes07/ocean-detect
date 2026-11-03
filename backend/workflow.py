"""
workflow.py — LangChain workflow for orca detection → social media post

This file chains together three LLM calls using LangChain's RunnableSequence.
Think of it like an assembly line: each station (RunnableLambda) receives
a dict of everything that came before, adds its own output, and passes it along.

Full chain (triggered only when the CNN detects an orca):

  audio file
      ↓
  [CNN in main.py]  ← confidence score
      ↓ (if detected)
  run_whale_voice()    → the named orca "speaks" — translates its call into English
      ↓
  run_generate_post()  → LLM writes a 2-paragraph Facebook post using that quote
      ↓
  run_moderate()           → second LLM reviews the post for accuracy and tone
      ↓
  run_post_to_facebook()   → posts approved final_post to the Facebook Page via Graph API
      ↓
  facebook_post_id (or None if moderation rejected or posting skipped)

Each step receives the full dict from all previous steps (via {**inputs, ...}),
so every LLM has access to all upstream context.

LangChain note:
  RunnableLambda wraps a plain Python function as a chainable step.
  The | operator pipes them: output of step N becomes input of step N+1.
  This is the same concept as Unix pipes: cat file.wav | detect | voice | post | moderate

POST /workflow
  Input:  audio file (multipart upload) + query params (orca_name, call_type, location)
  Output: full chain result — detected, confidence, whale quote, post draft,
          moderation decision, and final_post (null if rejected)
"""

import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil, tempfile

import requests

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

router = APIRouter()


# ─── LLM instances ────────────────────────────────────────────────────────────
#
# We create the LLM objects once at module load, not inside each function call.
# This avoids re-initializing the client on every request.
#
# Two separate instances with different temperatures:
#   llm           → temperature=0.8: creative, varied whale voice and post writing
#   moderator_llm → temperature=0.2: consistent, deterministic moderation decisions
#
# Using Claude Haiku: fastest and cheapest Anthropic model, good for short outputs.
# (Claude Sonnet/Opus would be higher quality but overkill for 2-3 sentence outputs.)
#
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.8,   # higher = more creative and varied responses
    max_tokens=512,
)

moderator_llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.2,   # lower = more consistent, predictable decisions
    max_tokens=256,
)


# ─── Orca personalities and call types ────────────────────────────────────────
#
# These give the LLM richer context so its responses are grounded in real SRKW facts.
# The personalities are based on documented individual orca histories.
#
ORCA_PERSONALITIES = {
    "Granny":    "the eldest matriarch, wise and calm, speaks with quiet authority",
    "Tahlequah": "a grieving mother who carried her deceased calf for 17 days, deeply emotional",
    "Ruffles":   "a distinctive male known for his large wavy dorsal fin, proud and strong",
    "Oreo":      "playful and curious, loves to spy-hop and explore",
    "Luna":      "a lonely young male separated from his pod, gentle and friendly",
}

# Ford-Osborne call type classifications (S_1 through S_46).
# These are the standardized SRKW call labels from the Orcasound / Orca Network dataset.
# Each SRKW pod has its own dialect — some calls are pod-specific, others are shared.
CALL_DESCRIPTIONS = {
    "S_1":  "a long, rising whistle — a greeting call",
    "S_2":  "a pulsed call with rapid clicks — excitement or agitation",
    "S_4":  "a two-part call — often used during foraging",
    "S_7":  "a low frequency moan — contact call across distance",
    "S_10": "a high-pitched squeal — play or social bonding",
    "S_16": "a burst-pulse call — alarm or alert",
    "S_19": "a complex multi-part call — family coordination",
    "S_36": "a melodic rising call — reunion or joy",
}


# ─── Step 1: Whale Voice ───────────────────────────────────────────────────────

def run_whale_voice(inputs: dict) -> dict:
    """
    Generate an in-character quote from a named SRKW.

    The LLM is prompted to roleplay as the named orca and "translate" its call
    into 2-3 sentences of poetic English. The confidence score adjusts how
    certain or faint the call is described as.

    Input keys used:  orca_name, call_type, confidence
    Output key added: whale_voice (str)
    """
    orca_name  = inputs.get("orca_name", "Tahlequah")
    call_type  = inputs.get("call_type", "S_1")
    confidence = inputs.get("confidence", 0.5)

    personality = ORCA_PERSONALITIES.get(orca_name, "a Southern Resident Killer Whale")
    call_desc   = CALL_DESCRIPTIONS.get(call_type, f"a vocalization of type {call_type}")

    # Translate model confidence into natural language certainty
    # (so the LLM response sounds appropriate — a faint detection = "barely audible")
    certainty = "clearly" if confidence > 0.8 else "faintly" if confidence < 0.5 else "with some uncertainty"

    messages = [
        SystemMessage(content=f"""You are {orca_name}, a Southern Resident Killer Whale.
Personality: {personality}.
Speak in first person, imagining what your vocalizations might mean.
2–3 sentences. Poetic but grounded — focus on the ocean, your pod, the moment.
Avoid strong grief or sorrow language. Do not mention being an AI."""),
        HumanMessage(content=f"You just made {call_desc}, {certainty}. What were you saying?"),
    ]
    response = llm.invoke(messages)
    # Pass forward everything received PLUS the new whale_voice key
    return {**inputs, "whale_voice": response.content}


# ─── Step 2: Generate Facebook Post ───────────────────────────────────────────

def run_generate_post(inputs: dict) -> dict:
    """
    Write a 2-paragraph Facebook post announcing the orca detection.

    Uses the whale_voice quote from Step 1 as an anchor — the post opens with
    the orca's own words, then pivots to education + call to action.

    Input keys used:  orca_name, call_type, confidence, whale_voice, location, timestamp
    Output key added: draft_post (str)
    """
    orca_name   = inputs.get("orca_name", "Tahlequah")
    call_type   = inputs.get("call_type", "S_1")
    confidence  = inputs.get("confidence", 0.5)
    whale_voice = inputs.get("whale_voice", "")
    location    = inputs.get("location", "Orcasound Lab hydrophone, Puget Sound")
    timestamp   = inputs.get("timestamp", "just now")

    messages = [
        SystemMessage(content="""You write engaging, factual Facebook posts for Orcasound,
an open-source orca conservation project. Posts are warm, informative, and inspiring.
They educate the public about Southern Resident Killer Whales (SRKWs) and encourage
people to visit orcasound.net to listen live. Never make up facts."""),
        HumanMessage(content=f"""Write a 2-paragraph Facebook post about this orca detection:

- Orca: {orca_name}
- Call type: {call_type} ({CALL_DESCRIPTIONS.get(call_type, 'vocalization')})
- Detection confidence: {confidence:.0%}
- Location: {location}
- Time: {timestamp}
- In {orca_name}'s own words: "{whale_voice}"

Paragraph 1: Announce the detection excitedly. Include the quote, but frame it clearly
as a poetic interpretation — use phrasing like "if she could speak" or "we imagine her
saying" so readers understand it is creative, not a literal translation.
Paragraph 2: Brief educational note about SRKWs and a call to action to visit orcasound.net."""),
    ]
    response = llm.invoke(messages)
    return {**inputs, "draft_post": response.content}


# ─── Step 3: Moderate ─────────────────────────────────────────────────────────

def run_moderate(inputs: dict) -> dict:
    """
    Review the draft post for factual accuracy, tone, and appropriateness.

    Uses a lower-temperature LLM (0.2) for consistent, deterministic decisions.
    The LLM is instructed to reply in a strict format so we can parse it reliably
    without fragile regex or JSON parsing.

    Checks:
      1. Factual accuracy — no invented statistics or false claims about orcas
      2. Appropriate tone — warm and educational, not sensationalist or clickbait
      3. No harmful content

    Input keys used:  draft_post
    Output keys added: approved (bool), moderation_reason (str), final_post (str | None)

    final_post is the approved draft, or None if rejected. Downstream steps (e.g.
    Facebook posting) should check approved before publishing.
    """
    draft_post = inputs.get("draft_post", "")

    messages = [
        SystemMessage(content="""You are a content moderator for Orcasound, an orca conservation nonprofit.
Review Facebook posts for:
1. Factual accuracy — no invented statistics or false claims about orca behavior or biology
2. Appropriate tone — warm and educational, not sensationalist
3. No harmful content

Clearly labeled creative or imaginative content (e.g. "if she could speak", "we imagine")
is acceptable and does NOT count as a factual claim. Approve posts that are warm,
educational, and clearly frame any whale "quotes" as poetic interpretation.

Respond with EXACTLY this format (no extra text before or after):
DECISION: APPROVED
REASON: <one sentence>

or:
DECISION: REJECTED
REASON: <one sentence explaining what's wrong>"""),
        HumanMessage(content=f"Review this post:\n\n{draft_post}"),
    ]
    response = moderator_llm.invoke(messages)
    text = response.content.strip()

    # Parse the structured response. Using line-by-line search instead of regex
    # so it degrades gracefully if the LLM adds extra whitespace.
    approved = "DECISION: APPROVED" in text
    reason = ""
    for line in text.splitlines():
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
            break  # stop at first REASON line

    return {
        **inputs,
        "approved": approved,
        "moderation_reason": reason,
        "final_post": inputs["draft_post"] if approved else None,
    }


# ─── Step 4: Post to Facebook ─────────────────────────────────────────────────

def run_post_to_facebook(inputs: dict) -> dict:
    """
    Post the approved final_post to a Facebook Page via the Graph API.

    This step is a no-op in two cases:
      - Moderation rejected the post (final_post is None) → skips silently
      - FACEBOOK_PAGE_TOKEN or FACEBOOK_PAGE_ID not set in .env → skips with a warning

    Facebook Graph API:
      POST https://graph.facebook.com/{page_id}/feed
        ?message={text}
        &access_token={page_access_token}

    The Page Access Token is different from your personal token or App ID.
    Get it from: developers.facebook.com → Graph API Explorer → select your Page →
    add pages_manage_posts permission → Generate Access Token.

    Set in .env:
      FACEBOOK_PAGE_TOKEN=EAABsbCS...   (100+ char string from Graph API Explorer)
      FACEBOOK_PAGE_ID=123456789        (numeric Page ID, found in Page's About section)

    Input keys used:  final_post, approved
    Output keys added: facebook_post_id (str | None), facebook_error (str | None)
    """
    # Skip if moderation rejected the post
    if not inputs.get("approved") or not inputs.get("final_post"):
        return {**inputs, "facebook_post_id": None, "facebook_error": "Skipped — post not approved"}

    page_token = os.environ.get("FACEBOOK_PAGE_TOKEN")
    page_id    = os.environ.get("FACEBOOK_PAGE_ID")

    # Skip gracefully if credentials aren't configured yet
    if not page_token or not page_id:
        return {
            **inputs,
            "facebook_post_id": None,
            "facebook_error": "Skipped — FACEBOOK_PAGE_TOKEN or FACEBOOK_PAGE_ID not set in .env",
        }

    # Call the Facebook Graph API to publish the post
    url = f"https://graph.facebook.com/{page_id}/feed"
    response = requests.post(url, data={
        "message": inputs["final_post"],
        "access_token": page_token,
    })

    if response.status_code == 200:
        post_id = response.json().get("id")
        print(f"Posted to Facebook: {post_id}")
        return {**inputs, "facebook_post_id": post_id, "facebook_error": None}
    else:
        # Return the error but don't raise — the post was written, just not published
        error = response.json().get("error", {}).get("message", response.text)
        print(f"Facebook post failed: {error}")
        return {**inputs, "facebook_post_id": None, "facebook_error": error}


# ─── LangChain Pipeline ────────────────────────────────────────────────────────
#
# The | operator chains RunnableLambdas into a RunnableSequence.
# Each step receives the full output dict of the previous step.
# This is equivalent to:
#   result = run_moderate(run_generate_post(run_whale_voice(inputs)))
# but expressed as a declarative pipeline that LangChain can trace, stream, and extend.
#
orca_workflow: RunnableSequence = (
    RunnableLambda(run_whale_voice)
    | RunnableLambda(run_generate_post)
    | RunnableLambda(run_moderate)
    | RunnableLambda(run_post_to_facebook)
)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/workflow")
async def run_workflow(
    file: UploadFile = File(...),
    orca_name: str = "Tahlequah",
    call_type: str = "S_1",
    location: str = "Orcasound Lab hydrophone, Puget Sound",
    force: bool = False,  # set ?force=true to skip detection and always run the chain
):
    """
    Full pipeline: upload audio → CNN detect → LangChain chain → response.

    If the CNN doesn't detect an orca (confidence ≤ 0.3), the workflow stops
    immediately and returns {"detected": false} without calling any LLMs.

    If detected, runs the full orca_workflow chain and returns all intermediate
    outputs so callers can inspect each step (not just the final post).

    Query params:
      orca_name  — which SRKW to voice (must be in ORCA_PERSONALITIES, or any name works)
      call_type  — Ford-Osborne call type label (e.g. "S_19")
      location   — hydrophone location for the Facebook post

    Example curl:
      curl -X POST http://localhost:8000/workflow \\
        -F "file=@clip.wav" \\
        -F "orca_name=Tahlequah" \\
        -F "call_type=S_19" \\
        -F "location=Orcasound Lab hydrophone, Puget Sound"
    """
    # Import here to avoid circular imports — workflow.py is imported by main.py,
    # so a top-level import of main.py would create a cycle.
    from backend.main import load_audio_window, audio_to_specgram, model, device

    import torch
    import torch.nn.functional as F
    from datetime import datetime, timezone

    # Save the uploaded file to a temp path so soundfile can read it
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        audio = load_audio_window(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=400, content={"error": f"Could not read audio: {e}"})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ── CNN Detection ──────────────────────────────────────────────────────────
    import numpy as np
    specgram = audio_to_specgram(audio)
    x = torch.tensor(specgram).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,71,1020]
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence = probs[0][1].item()   # orca class probability
        detected = confidence > 0.3       # low threshold → high recall

    # Short-circuit: don't burn LLM tokens if no orca was detected.
    # ?force=true bypasses this check for testing the full chain end-to-end.
    if not detected and not force:
        return JSONResponse(content={
            "detected": False,
            "confidence": round(confidence, 4),
            "message": "No orca detected — workflow not triggered.",
        })

    # ── LangChain Workflow ─────────────────────────────────────────────────────
    # Human-readable timestamp for the Facebook post
    timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # invoke() runs the full chain synchronously and returns the final dict,
    # which contains outputs from ALL steps (each step merges its results with {**inputs})
    result = orca_workflow.invoke({
        "orca_name": orca_name,
        "call_type": call_type,
        "confidence": confidence,
        "location": location,
        "timestamp": timestamp,
    })

    # Return all intermediate and final outputs for full transparency
    return JSONResponse(content={
        "detected": True,
        "confidence": round(confidence, 4),
        "orca_name": orca_name,
        "call_type": call_type,
        "whale_voice": result["whale_voice"],              # Step 1 output
        "draft_post": result["draft_post"],                # Step 2 output
        "approved": result["approved"],                    # Step 3 decision
        "moderation_reason": result["moderation_reason"],  # Step 3 reason
        "final_post": result["final_post"],                # None if rejected
        "facebook_post_id": result.get("facebook_post_id"),  # Step 4 — post ID if published
        "facebook_error": result.get("facebook_error"),      # Step 4 — error or skip reason
    })