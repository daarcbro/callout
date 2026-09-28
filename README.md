# CALLOUT

> **Real-time multilingual AI voice translator and tactical squad intelligence for competitive gaming.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AssemblyAI](https://img.shields.io/badge/AssemblyAI-Streaming%20v2-blue?style=flat)](https://www.assemblyai.com)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%2070B-orange?style=flat)](https://groq.com)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%20Shaders-black?style=flat&logo=three.js)](https://threejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚡ What is Callout?

In competitive online games (Valorant, CS2, Apex Legends, Rainbow Six Siege), split-second communication determines victory. However, international matchmaking often teams players who speak different languages.

**Callout** eliminates the language barrier:
- **Speak naturally in your native language** (Hindi, Hinglish, Spanish, French, Japanese, German, etc.).
- **Teammates hear the callout in their chosen language in under 1 second.**
- **Tactical gaming context**: General translators produce slow, polite sentences. Callout uses low-latency LLMs fine-tuned to translate gaming slang (*"bhai peeche se aa raha hai"* ➔ *"Flank behind you!"*).
- **Private Tactical AI Agent**: Hands-free private assistant commands (*"Agent, repeat"*, *"Agent, summary"*) keep your squad debriefed without spamming teammate comms.

---

## 🏗️ System Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     Player A (Speaker)                      │
   │  Mic Input  ──>  16 kHz PCM16 Stream  ──>  WebSocket Hub    │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │    AssemblyAI Streaming      │
                  │   Universal-3.5 Real-Time    │
                  │ (Hinglish/Code-Switching)    │
                  └───────────────┬──────────────┘
                                  │ Turn Completed (end_of_turn)
                                  ▼
                  ┌──────────────────────────────┐
                  │          Groq LLM            │
                  │   Tactical Gaming Engine     │
                  │   (Sub-200ms Translation)    │
                  └───────┬──────────────┬───────┘
                          │              │
        Teammate Translation             │ Private Assistant Request
                          ▼              ▼
   ┌─────────────────────────────┐   ┌─────────────────────────────┐
   │      Player B (Listener)    │   │      Player A (Private)     │
   │  Real-Time Tactical Feed    │   │  "Agent, repeat" / summary  │
   │  Audio: Browser TTS or      │   │  Private Audio Confirmation │
   │  ElevenLabs Flash Stream    │   │  (Never sent to squad)      │
   └─────────────────────────────┘   └─────────────────────────────┘
```

---

## 🌐 Application Pages & Routes

| Route | Interface | Description |
|---|---|---|
| `/` | **Interactive 3D Presentation Deck** (`index.html`) | Full-screen GPU-accelerated particle presentation featuring a 5-arm spinning galaxy, streamline fluid flow transitions, interactive feature dust cloud orb, 3D connected globe, room creator/joiner, and squad debrief thank you slide. |
| `/room` | **Dedicated Tactical Comms HUD** (`room.html`) | Clean, distraction-free voice terminal with live mic monitoring, push-to-talk / continuous speech, real-time translation feed, squad roster, and private voiceagent log. |
| `/health` | **System Status API** | Returns active rooms, connected players, and live AI model statuses. |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+**
- **AssemblyAI API Key** ([Free signup at AssemblyAI](https://www.assemblyai.com))
- **Groq API Key** ([Free signup at Groq](https://console.groq.com))
- *(Optional)* **ElevenLabs API Key** ([ElevenLabs](https://elevenlabs.io)) for hyper-realistic cloned gaming voices.

---

### 2. Installation

#### Windows (PowerShell / Command Prompt):
```powershell
# Clone the repository
git clone https://github.com/daarcbro/callout.git
cd callout

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create your .env configuration
copy .env.example .env
```

#### macOS / Linux:
```bash
# Clone the repository
git clone https://github.com/daarcbro/callout.git
cd callout

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your .env configuration
cp .env.example .env
```

---

### 3. Configure `.env`

Open `.env` in any text editor and fill in your API keys:

```ini
# Required: AssemblyAI Real-Time Speech-to-Text
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Required: Groq Ultra-Low Latency Inference
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional: High quality streamed voice output (default is free browser Web Speech)
TTS_PROVIDER=browser
# For ElevenLabs Flash:
# TTS_PROVIDER=elevenlabs
# ELEVENLABS_API_KEY=your_elevenlabs_key_here
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Server Settings
HOST=0.0.0.0
PORT=8000
```

---

### 4. Run the Server

```bash
python server.py
```

Now open **[http://localhost:8000](http://localhost:8000)** in your browser.

> ⚠️ **Microphone Note**: Modern web browsers strictly require **`localhost`** or a valid **`https://`** connection to access audio input devices.

---

## 🎮 How to Test (Step-by-Step)

### Step 1: Text-Based Fast Test (No mic needed)
1. Go to `http://localhost:8000/room?mode=create&name=Alpha&listen=en`.
2. Scroll to the bottom quick-test box.
3. Type a regional callout: `bhai peeche se aa raha hai bomb site B pe` and press Enter.
4. Verify Groq converts it immediately to a concise gaming callout: *"Enemy flanking behind on B site!"*.

### Step 2: Microphone & Speech-to-Text Verification
1. Click **"Connect & Unmute"** in the top bar.
2. Allow microphone access in your browser.
3. Speak clearly into your mic.
4. The live badge will display the real-time transcription and detected language.

### Step 3: Multi-Player Squad Test (Two Devices / Profiles)
1. **Host (Device 1)**: Click **"Copy Invite Link"** from inside the room.
2. **Teammate (Device 2 / Incognito)**: Open the link, enter a callsign (e.g., *Bravo*), and select their preferred listening language (e.g., *Hindi* or *Spanish*).
3. **Important**: Always use **headphones** during squad testing to prevent audio loops from speakers feeding back into the microphone.

---

## 🤖 Private Voice Agent Commands

Speak these trigger phrases into your microphone at any time. The AI processes your command privately and plays the response in your ears without broadcasting to your teammates:

| Command Phrase | Action | Use Case |
|---|---|---|
| **"Agent, repeat"** | Speaks the last received teammate callout. | When explosions or firefights drowned out teammate comms. |
| **"Agent, summary"** | Generates a 1-line tactical situational report of recent callouts. | After clutching or rotating between bombsites. |
| **"Agent, status"** | Reports active squad count and room ping. | Verifying connectivity before a round starts. |
| **"Agent, clear"** | Resets and clears your tactical feed display. | Starting a fresh match or half. |

---

## 📱 Testing on Two Physical Devices (LAN / Mobile)

Because browsers restrict microphone access to secure contexts, use a secure tunnel when testing between multiple physical devices:

```bash
# Using Cloudflare Tunnel (recommended, free & no account required):
cloudflared tunnel --url http://localhost:8000
```

Share the generated `https://xxxx.trycloudflare.com` URL with your teammates or phone. WebSockets will automatically upgrade to secure `wss://`.

---

## 📁 Repository Structure

```
├── index.html         # 3D WebGL particle slide deck (Galaxy, Dust Cloud Orb, Connected Globe, Thank You slide)
├── room.html          # High-performance tactical voice comms interface (16 kHz mic capture, real-time feed)
├── server.py          # FastAPI backend, WebSocket router, streaming AssemblyAI v2, Groq LLM, TTS hub
├── requirements.txt   # Python dependencies (fastapi, uvicorn, websockets, httpx, python-dotenv)
├── Procfile           # Production deployment process file
├── .env.example       # Example configuration template
└── README.md          # Project documentation
```

---

## 🛠️ Troubleshooting

| Issue | Likely Cause | Solution |
|---|---|---|
| **"Microphone access denied"** | Browser security restriction | Ensure you are on `http://localhost:8000` or an `https://` domain. Check browser mic permissions. |
| **"Translation failed" (Red badge)** | Invalid or expired Groq key | Check `GROQ_API_KEY` in `.env`. Ensure your model (`llama-3.3-70b-versatile`) is active on Groq. |
| **"Speech-to-text offline"** | AssemblyAI connection error | Verify `ASSEMBLYAI_API_KEY` in `.env` and that your account has streaming access enabled. |
| **Callouts appear on screen but no audio plays** | Browser audio autoplay policy | Check the **"Play callouts aloud"** checkbox and click anywhere in the room once to unlock audio playback. |
| **Audio feedback / echo loops** | Speakers playing into mic | Both players **must use headphones** or lower their speaker volume. |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Built with ❤️ for gamers and esports competitors worldwide.
