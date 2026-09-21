# Callout

Live squad voice comms translator. Each player talks in their own language; every teammate hears the callout in theirs, in about a second.

```
 Player A browser                                            Player B browser
 mic -> 16 kHz PCM16 --ws--> FastAPI hub                      <--ws-- text + voice
                              |  one AssemblyAI streaming session per player
                              v
                     AssemblyAI (universal-3-5-pro)   Hindi / Hinglish / English, code-switching
                              | finished turn (end_of_turn)
                              v
                     Groq LLM: gaming-aware translation into each listener's language
                              |
                    +---------+-----------+
                    v                     v
            browser voices (default)   ElevenLabs Flash (optional, streamed PCM)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# put your ASSEMBLYAI_API_KEY and GROQ_API_KEY in .env
python server.py
```

Open **http://localhost:8000** (microphones only work on `localhost` or https).

## Try it in this order

1. **Typed test.** Create a room, type `bhai peeche se aa raha hai` in the box at the bottom. You should see an English callout. This exercises Groq without the microphone.
2. **Mic test.** Speak Hindi. The line under "Your mic" shows exactly what the speech model heard, plus the language it detected. If that line is wrong, the problem is speech-to-text. If it is right but the callout is wrong, the problem is the translation model.
3. **Two players.** Open the invite link in a second browser profile or on a second device. Each person picks the language they want to hear. **Use headphones**, or one player's speakers will feed back into the other's mic.

## Voice agent

Say **"Agent, repeat"** to hear the last teammate callout again, or **"Agent, summary"** for a one-line tactical update of the last few callouts. Agent commands are answered privately and are never sent to teammates.

## Better voice (optional)

Default output uses the browser's built-in voices, which is free and needs no keys. For streamed studio-quality audio, set in `.env`:

```
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=...
```

If ElevenLabs errors mid-demo, the callout falls back to the browser voice automatically.

## Playing on two devices

Browsers block the mic on plain `http://` addresses other than localhost. Put the server behind https, for example:

```bash
cloudflared tunnel --url http://localhost:8000
```

then share the `https://...trycloudflare.com` link. WebSockets switch to `wss` automatically.

## Troubleshooting

| What you see | Likely cause |
|---|---|
| Red "Translation failed" in the feed | Read the message. Usually a wrong `GROQ_API_KEY`, a retired model (set `GROQ_MODEL`; the server prints your available models at startup), or a rate limit. |
| "Speech-to-text offline" | Check `ASSEMBLYAI_API_KEY` and that the account has streaming access. The terminal shows AssemblyAI's exact error. |
| Callouts arrive but nothing is spoken | Tick "Play callouts aloud", and click once in the page before the first callout (browsers block audio until you interact). |
| Hindi voice sounds wrong | Your OS may lack a Hindi voice. Use ElevenLabs, or listen in English. |

`GET /health` shows the active models and rooms.

## 2-minute demo script

1. (15s) Show the lobby. Create a room on a laptop, join from a phone.
2. (30s) Laptop speaks Hindi callouts; phone hears English. Show the big callout on screen.
3. (30s) Phone replies in English; laptop hears Hindi.
4. (20s) "Agent, repeat" and "Agent, summary".
5. (25s) Architecture slide: AssemblyAI streaming, Groq, one WebSocket per player, sub-second end to end.

## Files

- `server.py`: rooms, per-player streaming STT, translation, voice agent, TTS
- `index.html`: the whole frontend (lobby, comms screen, 16 kHz mic capture, audio playback)
