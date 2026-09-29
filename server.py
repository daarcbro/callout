"""
Callout - live squad voice comms translator.

Browser mic -> FastAPI (one WebSocket per player) -> AssemblyAI streaming STT
-> Groq LLM (gaming-aware translation) -> teammates' browsers (text + voice).

Run:  python server.py     then open http://localhost:8000
"""
import asyncio
import json
import logging
import os
import re
import secrets
import string
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from groq import AsyncGroq
import json

from assemblyai.streaming.v3 import (
    AsyncStreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("callout")
logging.getLogger("httpx").setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLYAI_KEY_TEST")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("STT_MODEL", "universal-3-5-pro")
STT_MODE = os.getenv("STT_MODE", "balanced")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "browser").lower()
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVEN_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
if TTS_PROVIDER == "elevenlabs" and not ELEVEN_KEY:
    log.warning("TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is missing - using browser voices.")
    TTS_PROVIDER = "browser"

MAX_PLAYERS = 6
HISTORY_LIMIT = 30
PLAYER_COLORS = 6  # the frontend maps color index -> hue

LANGS = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "ja": "Japanese", "ar": "Arabic", "tr": "Turkish",
}

# Languages whose scripts most browsers can't speak aloud.
# For these, we transliterate to Latin (romanize) before sending to browser TTS.
ROMANIZE_LANGS = {"hi", "ja", "ar"}

# Words the speech model should be primed to hear correctly.
KEYTERMS = [
    "one-shot", "flank", "push", "rotate", "revive", "camping", "catwalk", "diff",
    "clutch", "spawn", "headshot", "knocked", "smoke", "peek", "ult",
    "बंदा", "पीछे", "पुश", "रिवाइव", "कैंप", "वन शॉट", "फ्लैंक",
]

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Shared clients (created in lifespan)
# --------------------------------------------------------------------------- #
groq_client: Optional[AsyncGroq] = None
http: Optional[httpx.AsyncClient] = None
_background: set = set()


def spawn(coro) -> asyncio.Task:
    """create_task that keeps a reference so the task isn't garbage collected."""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task

# Simple in‑memory cache for translation results
_translation_cache: dict[tuple[str, str], str] = {}

def _cache_get(key: tuple[str, str]):
    return _translation_cache.get(key)

def _cache_set(key: tuple[str, str], value: str):
    _translation_cache[key] = value


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, http
    groq_client = AsyncGroq(api_key=GROQ_KEY)
    http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

    log.info("Callout starting")
    log.info("  STT   : AssemblyAI %s (%s)  key=%s", STT_MODEL, STT_MODE, "set" if ASSEMBLYAI_KEY else "MISSING")
    log.info("  LLM   : Groq %s  key=%s", GROQ_MODEL, "set" if GROQ_KEY else "MISSING")
    log.info("  Voice : %s", TTS_PROVIDER)
    if GROQ_KEY:
        try:
            models = await groq_client.models.list()
            ids = sorted(m.id for m in models.data)
            if GROQ_MODEL not in ids:
                log.warning("GROQ_MODEL '%s' is not in your Groq model list. Available: %s", GROQ_MODEL, ", ".join(ids))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not verify Groq model list: %r", e)
    yield
    await http.aclose()


app = FastAPI(lifespan=lifespan)

# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
GLOSSARY = (
    "Gaming terms: one-shot = enemy has almost no health; camping = staying hidden in one spot; "
    "flank = attacking from the side or behind; push = advance aggressively; rotate = move to another area; "
    "revive = pick up a downed teammate; diff = a much better player (slang); clutch = winning while outnumbered. "
    "Map or item names (for example 'catwalk') stay as they are."
)

FEW_SHOT_EN = [
    ("बंदा पीछे है", "Enemy behind you!"),
    ("bhai wo one shot hai push kar", "He's one-shot, push!"),
    ("revive kar do jaldi", "Need a revive, fast!"),
    ("वो कैंप कर रहा है", "He's camping!"),
    ("यार तू सुन नहीं रहा क्या", "Bro, are you even listening?"),
]


class TranslationError(Exception):
    pass


def _system_prompt(target_name: str) -> str:
    return (
        "You are a live esports comms interpreter for squad voice chat.\n"
        "The input is a speech transcript of what a teammate just said. It may be Hindi in Devanagari, "
        "Hinglish (Hindi typed in Roman letters), English, or another language, and it may mix them.\n"
        f"Rewrite it as a short, punchy tactical callout in {target_name}.\n"
        "Rules:\n"
        f"1. Reply in {target_name} only, in its normal script. Never copy the source language unless it is already {target_name}.\n"
        "2. Keep the exact meaning. Use natural gamer wording for that language.\n"
        "3. Ten words or fewer when possible.\n"
        f"4. If the input is already {target_name}, just clean it up and tighten it.\n"
        "5. Translate everything faithfully, including small talk. Never refuse and never explain.\n"
        "6. Output ONLY the callout. No quotes, labels, notes or extra lines.\n"
        f"{GLOSSARY}"
    )


async def translate(text: str, target: str) -> str:
    # Check cache first
    cache_key = (text, target)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    target_name = LANGS.get(target, "English")
    messages = [{"role": "system", "content": _system_prompt(target_name)}]
    if target == "en":
        for src, out in FEW_SHOT_EN:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": text})
    try:
        resp = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=80
            ),
            timeout=8,
        )
    except Exception as e:  # noqa: BLE001
        raise TranslationError(f"{type(e).__name__}: {e}") from e
    out = (resp.choices[0].message.content or "").strip()
    out = out.splitlines()[0].strip().strip("\"'“”") if out else ""
    if not out:
        raise TranslationError("the model returned an empty reply")
    _cache_set(cache_key, out)
    return out

async def translate_multi(text: str, targets: list[str]) -> dict:
    """Translate `text` into all `targets` languages using a single Groq request.
    Returns a mapping of language code -> translation.
    """
    langs_list = ", ".join(targets)
    system = (
        "You are a live esports comms interpreter. Provide translations for the following languages: "
        f"{langs_list}. Return a JSON object where each key is the language code and the value is the translated callout. Output only the JSON."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    try:
        resp = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=200
            ),
            timeout=8,
        )
    except Exception as e:  # noqa: BLE001
        raise TranslationError(f"{type(e).__name__}: {e}") from e
    content = resp.choices[0].message.content or ""
    try:
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Response is not a dict")
    except Exception as e:
        raise TranslationError(f"Failed to parse multi-translation JSON: {e}") from e
    return result

async def summarize(entries: list, target: str) -> str:
    target_name = LANGS.get(target, "English")
    lines = "\n".join(f"{e['from']}: {e['translations'].get(target) or e['original']}" for e in entries)
    resp = await asyncio.wait_for(
        groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": f"You are Callout, an elite esports squad tactical AI. Summarize recent squad comms into a crisp, high-impact tactical situation report in {target_name}. 25 words max. Focus on enemy positions, player status, and map objectives. Output only the briefing without preface."},
                {"role": "user", "content": lines},
            ],
            temperature=0.3,
            max_tokens=100,
        ),
        timeout=8,
    )
    return (resp.choices[0].message.content or "").strip()


async def romanize(text: str, source_lang: str) -> str:
    """Transliterate non-Latin text into phonetic Latin script for browser TTS."""
    cache_key = (text, f"_romanize_{source_lang}")
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    lang_name = LANGS.get(source_lang, source_lang)
    system = (
        f"You are a transliteration engine. Convert the following {lang_name} text "
        f"into phonetic Latin/Roman script so an English text-to-speech engine can "
        f"pronounce it naturally and be understood by a {lang_name} speaker.\n"
        f"Rules:\n"
        f"1. Output ONLY the romanized text. No translations, explanations, or quotes.\n"
        f"2. Use intuitive English-friendly spelling (e.g. 'sh' not 'ś', 'ch' not 'c̄').\n"
        f"3. Keep English loanwords and gaming terms as-is (push, one-shot, revive, etc).\n"
        f"4. Keep it short — same length as the input."
    )
    try:
        resp = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ], temperature=0.1, max_tokens=80
            ),
            timeout=6,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Romanization failed for %s: %r", source_lang, e)
        return text  # fallback: send the original text
    out = (resp.choices[0].message.content or "").strip()
    out = out.splitlines()[0].strip().strip("\"'“”") if out else text
    if not out:
        return text
    _cache_set(cache_key, out)
    log.info("Romanized [%s]: %s -> %s", source_lang, text, out)
    return out


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #
async def eleven_stream(text: str) -> AsyncIterator[bytes]:
    """Yield 16 kHz mono 16-bit PCM from ElevenLabs, always an even number of bytes."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
    headers = {"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json", "Accept": "audio/*"}
    payload = {"text": text, "model_id": ELEVEN_MODEL}
    carry = b""
    async with http.stream("POST", url, params={"output_format": "pcm_16000"}, headers=headers, json=payload) as r:
        if r.status_code != 200:
            body = (await r.aread()).decode("utf-8", "ignore")[:200]
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {body}")
        async for chunk in r.aiter_bytes(4096):
            data = carry + chunk
            if len(data) % 2:
                carry, data = data[-1:], data[:-1]
            else:
                carry = b""
            if data:
                yield data


# --------------------------------------------------------------------------- #
# Rooms and players
# --------------------------------------------------------------------------- #
class Player:
    def __init__(self, ws: WebSocket, name: str, room: "Room", listen: str, color: int):
        self.id = secrets.token_hex(3)
        self.ws = ws
        self.name = name
        self.room = room
        self.listen = listen
        self.color = color
        self._lock = asyncio.Lock()
        self.utterances: asyncio.Queue = asyncio.Queue()
        self.tts_q: asyncio.Queue = asyncio.Queue()
        self.stt: Optional[AsyncStreamingClient] = None
        self.stt_alive = False
        self.stt_error = ""
        self.stt_reconnecting = False
        self.last_stt_attempt = 0.0
        self.last_turn_order = -1
        self.last_activity = 0.0
        self.last_partial_text = ""
        self.last_partial_time = 0.0
        self.last_partial_lang = None
        self.closed = False

    async def send(self, obj: dict) -> None:
        if self.closed:
            return
        try:
            async with self._lock:
                await self.ws.send_json(obj)
        except Exception:  # socket already gone
            self.closed = True

    async def send_bytes(self, data: bytes) -> None:
        if self.closed:
            return
        try:
            async with self._lock:
                await self.ws.send_bytes(data)
        except Exception:
            self.closed = True

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "listen": self.listen, "color": self.color}


class Room:
    def __init__(self, room_id: str):
        self.id = room_id
        self.players: dict = {}
        self.history: list = []
        self._color_i = 0

    def next_color(self) -> int:
        c = self._color_i % PLAYER_COLORS
        self._color_i += 1
        return c

    async def broadcast(self, obj: dict, exclude: Optional[str] = None) -> None:
        await asyncio.gather(*(p.send(obj) for p in list(self.players.values()) if p.id != exclude))

    async def send_roster(self) -> None:
        await self.broadcast({"type": "roster", "players": [p.public() for p in self.players.values()]})


ROOMS: dict = {}
CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def new_room_code() -> str:
    while True:
        code = "".join(secrets.choice(CODE_CHARS) for _ in range(4))
        if code not in ROOMS:
            return code


def clean_room(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())[:8]


def clean_name(raw: str) -> str:
    return re.sub(r"[\x00-\x1f<>]", "", (raw or "")).strip()[:20]


# --------------------------------------------------------------------------- #
# Speech-to-text (one AssemblyAI streaming session per player)
# --------------------------------------------------------------------------- #
async def _open_stt(player: Player, params: StreamingParameters) -> bool:
    settled = asyncio.Event()  # set on Begin or on Error, whichever comes first
    player.stt_error = ""
    client = AsyncStreamingClient(StreamingClientOptions(api_key=ASSEMBLYAI_KEY, connect_timeout=4.0, max_connection_retries=1))

    async def on_begin(_c, _ev):
        player.stt_alive = True
        settled.set()

    async def on_turn(_c, ev):
        await handle_turn(player, ev)

    async def on_error(_c, err):
        player.stt_alive = False
        player.stt_error = str(getattr(err, "message", err))
        log.warning("[%s] AssemblyAI error: %s", player.name, player.stt_error)
        settled.set()

    async def on_term(_c, _ev):
        player.stt_alive = False
        log.info("[%s] AssemblyAI session ended", player.name)

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Error, on_error)
    client.on(StreamingEvents.Termination, on_term)

    await client.connect(params)
    try:
        await asyncio.wait_for(settled.wait(), timeout=6)
    except asyncio.TimeoutError:
        player.stt_alive = False
    if player.stt_alive:
        player.stt = client
        return True
    try:
        await client.disconnect()
    except Exception:
        pass
    return False


def _friendly_stt_error(err: str) -> str:
    if "401" in err or "403" in err:
        return f"AssemblyAI rejected the request ({err}). Check ASSEMBLYAI_API_KEY and that your account has streaming access."
    return err


async def start_stt(player: Player) -> bool:
    """Try full settings first; if an optional parameter was the problem, retry with a minimal config."""
    player.last_stt_attempt = time.monotonic()
    attempts = [
        StreamingParameters(
            speech_model=STT_MODEL, sample_rate=16000, mode=STT_MODE,
            language_detection=True, keyterms_prompt=KEYTERMS,
            min_turn_silence=350, max_turn_silence=700, min_end_of_turn_silence_when_confident=250,
        ),
        StreamingParameters(speech_model=STT_MODEL, sample_rate=16000, min_turn_silence=400),
    ]
    for i, params in enumerate(attempts):
        try:
            if await _open_stt(player, params):
                log.info("[%s] speech-to-text connected%s", player.name, " (minimal settings)" if i else "")
                return True
        except Exception as e:  # noqa: BLE001
            player.stt_error = repr(e)
            log.warning("[%s] STT connect failed: %r", player.name, e)
        err = player.stt_error
        if any(x in err for x in ("401", "403", "Errno", "timed out", "getaddrinfo")):
            break  # auth or network problem: a second attempt can't fix it
    player.stt_error = _friendly_stt_error(player.stt_error)
    return False


async def reconnect_stt(player: Player) -> None:
    if player.stt_reconnecting:
        return
    player.stt_reconnecting = True
    try:
        await player.send({"type": "status", "text": "Reconnecting speech-to-text..."})
        ok = await start_stt(player)
        await player.send({"type": "status", "text": "Speech-to-text is back." if ok else "", "stt": ok})
        if not ok:
            await player.send({"type": "error", "where": "stt", "message": player.stt_error or "Could not reconnect speech-to-text."})
    finally:
        player.stt_reconnecting = False


async def push_audio(player: Player, data: bytes) -> None:
    if not player.stt_alive or player.stt is None:
        if time.monotonic() - player.last_stt_attempt > 5 and not player.stt_reconnecting:
            spawn(reconnect_stt(player))
        return
    try:
        await player.stt.stream(data)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] stream error: %r", player.name, e)
        player.stt_alive = False


GHOST_NOISE_RE = re.compile(
    r"^[\s\.\,\?\!\-\—\_]+$|^\[.*\]$|^(?:thank\s+you|you|um|uh|ah|yeah|shh|hmm|okay|ok)\.?$",
    re.I
)


async def _partial_watchdog(player: Player, text: str, captured_time: float, lang: Optional[str]) -> None:
    await asyncio.sleep(1.2)
    if getattr(player, "last_partial_time", 0) == captured_time and getattr(player, "last_partial_text", "") == text and not player.closed:
        log.info("[%s] silence watchdog finalizing turn: %s", player.name, text)
        player.last_partial_text = ""
        player.last_partial_time = 0.0
        player.utterances.put_nowait((text, lang))


async def handle_turn(player: Player, ev) -> None:
    text = (ev.transcript or "").strip()
    if not text or GHOST_NOISE_RE.match(text):
        return
    if not ev.end_of_turn:
        player.last_partial_text = text
        player.last_partial_time = time.monotonic()
        player.last_partial_lang = getattr(ev, "language_code", None)
        spawn(_partial_watchdog(player, text, player.last_partial_time, player.last_partial_lang))
        await player.send({"type": "partial", "text": text})
        now = time.monotonic()
        if now - player.last_activity > 0.6:
            player.last_activity = now
            await player.room.broadcast({"type": "activity", "id": player.id}, exclude=player.id)
        return
    player.last_partial_text = ""
    player.last_partial_time = 0.0
    order = getattr(ev, "turn_order", None)
    if order is not None and order == player.last_turn_order:
        return  # duplicate end-of-turn for the same turn
    player.last_turn_order = order if order is not None else player.last_turn_order
    lang = getattr(ev, "language_code", None)
    log.info("[%s] heard (%s): %s", player.name, lang or "?", text)
    player.utterances.put_nowait((text, lang))


# --------------------------------------------------------------------------- #
# Callout pipeline
# --------------------------------------------------------------------------- #
AGENT_RE = re.compile(
    r"^\W*(?:hey\s+|ok\s+|okay\s+|yo\s+)?(?:callout|call\s*out|agent|कॉलाउट|एजेंट|एजेन्ट)(?:[\s,:\-—]+(.*)|$)",
    re.I | re.S,
)
REPEAT_RE = re.compile(r"repeat|again|say that|once more|last call|dobara|phir se|दोबारा|फिर से|क्या बोला", re.I)
SUMMARY_RE = re.compile(r"summar|status|situation|recap|update|brief|what happened|kya hua|kya chal raha|सारांश|हाल", re.I)


async def ask_tactical_copilot(command: str, entries: list, target: str) -> str:
    target_name = LANGS.get(target, "English")
    lines = (
        "\n".join(f"{e['from']}: {e['translations'].get(target) or e['original']}" for e in entries[-8:])
        if entries
        else "No prior squad comms."
    )
    system = (
        f"You are Callout, an elite AI tactical gaming copilot assisting a squad in live voice chat.\n"
        f"The player asked: '{command}'\n"
        f"Recent squad callouts:\n{lines}\n"
        f"Rules:\n"
        f"1. Answer in {target_name} concisely in 20 words or fewer.\n"
        f"2. Be sharp, tactical, and direct. Sound like an esports coach or tactical AI assistant.\n"
        f"3. Output ONLY the response, no conversational filler or quotes."
    )
    try:
        resp = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": command},
                ],
                temperature=0.3,
                max_tokens=80,
            ),
            timeout=8,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = out.splitlines()[0].strip().strip("\"'“”") if out else ""
        return out or "Callout AI standing by."
    except Exception as e:
        log.warning("Copilot answer failed: %s", e)
        return "Callout standing by. Say 'Hey Callout, summarize' or 'repeat'."


async def utterance_worker(player: Player) -> None:
    """Process one player's utterances in order so callouts never arrive shuffled."""
    while True:
        text, lang = await player.utterances.get()
        clean = (text or "").strip()
        if not clean or GHOST_NOISE_RE.match(clean) or len(clean) < 2:
            continue
        try:
            await process_utterance(player, clean, lang)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] pipeline error", player.name)
            await player.send({"type": "error", "where": "pipeline", "message": repr(e)})


async def process_utterance(player: Player, text: str, lang: Optional[str]) -> None:
    room = player.room
    await player.send({"type": "final", "text": text, "lang": lang})

    m = AGENT_RE.match(text)
    if m:
        await run_agent(player, m.group(1).strip() if m.group(1) else "")
        return

    others = [p for p in room.players.values() if p.id != player.id]
    solo = not others
    targets = [player] if solo else others
    target_langs = list(dict.fromkeys(p.listen for p in targets))

    started = time.monotonic()
    try:
        translations = await translate_multi(text, target_langs)
    except Exception as e:
        log.warning("[%s] multi-translation failed: %s", player.name, e)
        # fallback to per-language translation
        results = await asyncio.gather(*(translate(text, l) for l in target_langs), return_exceptions=True)
        translations = {}
        for l, r in zip(target_langs, results):
            if isinstance(r, Exception):
                log.warning("[%s] translation to %s failed: %s", player.name, l, r)
                await player.send({"type": "error", "where": "translate", "message": f"Translation failed: {r}"})
            else:
                translations[l] = r
    if not translations:
        return
    log.info("[%s] translated in %d ms: %s", player.name, (time.monotonic() - started) * 1000, translations)

    room.history.append({"from_id": player.id, "from": player.name, "original": text,
                         "translations": translations, "ts": time.time()})
    del room.history[:-HISTORY_LIMIT]

    for p in targets:
        t = translations.get(p.listen)
        if t is None:
            continue
        await p.send({"type": "callout", "from": player.name, "from_id": player.id, "color": player.color,
                      "original": text, "translated": t, "lang": p.listen, "detected": lang, "solo": solo})
        if not solo:
            p.tts_q.put_nowait(t)


async def run_agent(player: Player, command: str) -> None:
    room = player.room
    cmd = (command or "").lower().strip()

    async def reply(text: str, label: str, speak: bool = True) -> None:
        await player.send({"type": "agent", "label": label, "text": text})
        if speak:
            player.tts_q.put_nowait(text)

    if not cmd:
        await reply("Callout AI online. Say: summarize, repeat, or ask a tactical question.", "AI Copilot")
        return

    if REPEAT_RE.search(cmd):
        pool = [e for e in room.history if e["from_id"] != player.id] or room.history
        if not pool:
            await reply("Nothing to repeat yet.", "Repeat")
            return
        entry = pool[-1]
        text = entry["translations"].get(player.listen)
        if text is None:
            text = await translate(entry["original"], player.listen)
            entry["translations"][player.listen] = text
        await reply(text, f"Repeat from {entry['from']}")
    elif SUMMARY_RE.search(cmd):
        recent = room.history[-10:]
        if not recent:
            await reply("No squad callouts recorded yet.", "Tactical Recap")
            return
        try:
            await reply(await summarize(recent, player.listen), "Tactical Recap")
        except Exception as e:  # noqa: BLE001
            await player.send({"type": "error", "where": "agent", "message": f"Summary failed: {e}"})
    else:
        try:
            ans = await ask_tactical_copilot(command, room.history, player.listen)
            await reply(ans, "Tactical AI")
        except Exception:
            await reply("Callout standing by.", "Tactical AI", speak=False)


async def _browser_speak(player: Player, text: str) -> None:
    """Send text to the browser for speech synthesis, romanizing if needed."""
    speak_text, speak_lang = text, player.listen
    if player.listen in ROMANIZE_LANGS:
        try:
            speak_text = await romanize(text, player.listen)
            speak_lang = "en"  # English voice can pronounce the romanized text
        except Exception:  # noqa: BLE001
            pass  # worst case: send original text with original lang
    await player.send({"type": "speak", "text": speak_text, "lang": speak_lang})


async def tts_worker(player: Player) -> None:
    """Speak queued text to one player, one item at a time."""
    while True:
        text = await player.tts_q.get()
        sent_audio = False
        try:
            if TTS_PROVIDER == "elevenlabs":
                async for chunk in eleven_stream(text):
                    sent_audio = True
                    await player.send_bytes(chunk)
            else:
                await _browser_speak(player, text)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] TTS failed: %r", player.name, e)
            if not sent_audio:  # fall back to the browser voice so the callout is still heard
                await _browser_speak(player, text)


# --------------------------------------------------------------------------- #
# HTTP + WebSocket endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/room")
async def room_page():
    return FileResponse(BASE_DIR / "room.html")


@app.get("/health")
async def health():
    return JSONResponse({
        "stt": {"model": STT_MODEL, "key": bool(ASSEMBLYAI_KEY)},
        "llm": {"model": GROQ_MODEL, "key": bool(GROQ_KEY)},
        "tts": TTS_PROVIDER,
        "rooms": {rid: len(r.players) for rid, r in ROOMS.items()},
    })


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    mode = q.get("mode", "create")
    name = clean_name(q.get("name", "")) or "Player"
    listen = q.get("listen", "en")
    if listen not in LANGS:
        listen = "en"

    if mode == "join":
        room_id = clean_room(q.get("room", ""))
        room = ROOMS.get(room_id)
        if room is None:
            await ws.send_json({"type": "fatal", "message": f"No room with code {room_id or '(empty)'}. Check the code or create a new room."})
            await ws.close()
            return
    else:
        room_id = new_room_code()
        room = ROOMS.setdefault(room_id, Room(room_id))
    if len(room.players) >= MAX_PLAYERS:
        await ws.send_json({"type": "fatal", "message": f"Room {room.id} is full ({MAX_PLAYERS} players)."})
        await ws.close()
        return

    player = Player(ws, name, room, listen, room.next_color())
    room.players[player.id] = player
    workers = [spawn(utterance_worker(player)), spawn(tts_worker(player))]
    log.info("[%s] joined room %s (listens in %s)", name, room.id, listen)

    try:
        stt_ok = await start_stt(player) if ASSEMBLYAI_KEY else False
        if not ASSEMBLYAI_KEY:
            player.stt_error = "ASSEMBLYAI_API_KEY is not set in .env"
        await player.send({
            "type": "hello", "you": player.id, "room": room.id, "tts": TTS_PROVIDER, "stt": stt_ok,
            "langs": LANGS, "color": player.color, "max_players": MAX_PLAYERS,
        })
        if not stt_ok:
            await player.send({"type": "error", "where": "stt",
                               "message": f"Speech-to-text isn't connected: {player.stt_error or 'unknown error'}. You can still test translation with the text box."})
        await room.send_roster()

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await push_audio(player, msg["bytes"])
            elif msg.get("text"):
                await handle_control(player, msg["text"])
    except Exception as e:  # noqa: BLE001
        log.info("[%s] connection ended: %r", name, e)
    finally:
        # Cleanup runs in detached tasks so it still completes if this handler was cancelled.
        player.closed = True
        for w in workers:
            w.cancel()
        if player.stt is not None:
            spawn(_close_stt(player.stt))
        room.players.pop(player.id, None)
        log.info("[%s] left room %s", name, room.id)
        if not room.players:
            ROOMS.pop(room.id, None)
        else:
            spawn(room.send_roster())


async def _close_stt(client) -> None:
    try:
        await client.disconnect(terminate=True)
    except Exception:
        pass


async def handle_control(player: Player, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except ValueError:
        return
    kind = msg.get("type")
    if kind == "set_listen" and msg.get("lang") in LANGS:
        player.listen = msg["lang"]
        await player.room.send_roster()
    elif kind == "text":
        text = str(msg.get("text", "")).strip()[:300]
        if text:
            player.utterances.put_nowait((text, None))
    elif kind == "summarize":
        recent = player.room.history[-10:]
        if not recent:
            await player.send({"type": "agent", "label": "Tactical Recap", "text": "No squad comms recorded yet."})
            player.tts_q.put_nowait("No squad comms recorded yet.")
        else:
            try:
                summary_text = await summarize(recent, player.listen)
                await player.send({"type": "agent", "label": "Tactical Recap", "text": summary_text})
                player.tts_q.put_nowait(summary_text)
            except Exception as e:
                await player.send({"type": "error", "where": "agent", "message": f"Summary failed: {e}"})
    elif kind == "repeat":
        pool = [e for e in player.room.history if e["from_id"] != player.id] or player.room.history
        if not pool:
            await player.send({"type": "agent", "label": "Repeat", "text": "Nothing to repeat yet."})
            player.tts_q.put_nowait("Nothing to repeat yet.")
        else:
            entry = pool[-1]
            text = entry["translations"].get(player.listen)
            if text is None:
                text = await translate(entry["original"], player.listen)
                entry["translations"][player.listen] = text
            await player.send({"type": "agent", "label": f"Repeat from {entry['from']}", "text": text})
            player.tts_q.put_nowait(text)
    elif kind == "agent_ask":
        query = str(msg.get("text", "")).strip()[:150]
        if query:
            await run_agent(player, query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
