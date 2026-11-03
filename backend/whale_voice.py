"""
whale_voice.py — FastAPI endpoint for orca voice generation

POST /whale-voice
  call_type  : Ford-Osborne call type, e.g. "S_1" or "S_19"
  orca_name  : Named SRKW, e.g. "Granny" or "Tahlequah"
  confidence : Detection confidence score (0.0–1.0)

Returns a short in-character response as the named orca "translating" its call.
"""

import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

router = APIRouter()

# Known Southern Resident Killer Whales with brief personality notes
ORCA_PERSONALITIES = {
    "Granny": "the eldest matriarch, wise and calm, speaks with quiet authority",
    "Tahlequah": "a grieving mother who carried her deceased calf for 17 days, deeply emotional",
    "Ruffles": "a distinctive male known for his large wavy dorsal fin, proud and strong",
    "Oreo": "playful and curious, loves to spy-hop and explore",
    "Luna": "a lonely young male separated from his pod, gentle and friendly",
}

# Ford-Osborne call type descriptions (subset of S_1–S_46)
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


class WhaleVoiceRequest(BaseModel):
    call_type: str   # e.g. "S_1"
    orca_name: str   # e.g. "Granny"
    confidence: float  # 0.0–1.0


@router.post("/whale-voice")
async def whale_voice(req: WhaleVoiceRequest):
    personality = ORCA_PERSONALITIES.get(
        req.orca_name,
        "a Southern Resident Killer Whale, wise and social"
    )
    call_desc = CALL_DESCRIPTIONS.get(
        req.call_type,
        f"a vocalization of type {req.call_type}"
    )

    # Vary the certainty language based on confidence score
    if req.confidence > 0.8:
        certainty = "clearly and confidently"
    elif req.confidence > 0.5:
        certainty = "with some uncertainty"
    else:
        certainty = "faintly, barely audible"

    system_prompt = f"""You are {req.orca_name}, a Southern Resident Killer Whale.
Your personality: {personality}.
You speak in first person as this orca, translating your own vocalizations into English.
Keep responses to 2–3 sentences. Be poetic, emotional, and true to the ocean.
Do not mention being an AI or a language model."""

    human_prompt = f"""You just made {call_desc}, {certainty}.
What were you saying? Translate your call into words."""

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.85,
        max_tokens=256,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)

    return JSONResponse(content={
        "orca_name": req.orca_name,
        "call_type": req.call_type,
        "confidence": req.confidence,
        "voice": response.content,
    })