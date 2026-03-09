# Arrow Backend 🏹

The backend powers Arrow's full voice pipeline — speech recognition, LLM reasoning, live web search, document intelligence, and text-to-speech. It runs as a LiveKit Agent Worker on AWS EC2.

---

## What It Does

- Receives audio from the user's phone over WebRTC via LiveKit
- Detects end of speech with Silero VAD
- Transcribes speech in 16+ Indian languages via Amazon Transcribe
- Reasons and generates responses using Claude Haiku on AWS Bedrock
- Searches live mandi prices, scheme updates, and news via Exa MCP
- Reads uploaded documents (PDF, image, Excel, CSV) via Sarvam Document Intelligence
- Speaks the response back using Amazon Polly Neural TTS

---

## Repositories

| Part | Repo |
|---|---|
| Backend (this repo) | `ArrowBackend` |
| Frontend | `ArrowFrontend` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LiveKit Agents SDK (Python) |
| VAD | Silero VAD |
| STT | Amazon Transcribe Streaming |
| LLM | Claude Haiku via AWS Bedrock |
| TTS | Amazon Polly Neural |
| Web Search | Exa Search API via MCP |
| Document OCR | Sarvam Document Intelligence |
| Excel / CSV | openpyxl, stdlib csv |
| Process Manager | systemd |
| Server | AWS EC2 t3.medium |

---

## Project Structure

```
ArrowBackend/
├── agents.py          # Main LiveKit agent — full voice pipeline
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── README.md
```

---

## Environment Variables

Create `/etc/agents.env` on your EC2 (never commit secrets):

```env
# LiveKit
LIVEKIT_URL=wss://your-livekit-server.nip.io
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# AWS
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=ap-south-1

# Sarvam
SARVAM_API_KEY=your_sarvam_api_key

# Exa
EXA_API_KEY=your_exa_api_key
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A running LiveKit Server (see LiveKit EC2 setup below)
- AWS IAM user with permissions for Transcribe, Polly, and Bedrock

### Install

```bash
git clone https://github.com/19Doors/ArrowBackend
cd ArrowBackend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run locally

```bash
cp .env.example .env
# Fill in your credentials
source .env
python agents.py dev
```

---

## Agent Pipeline (per session)

Every time a user joins a LiveKit room, the agent runs this pipeline:

```
User speaks
  └── Silero VAD          → detects end of utterance
  └── Amazon Transcribe   → speech to text (16+ Indian languages)
  └── Claude Haiku        → understands query, decides action
        ├── Exa MCP       → live search (prices, schemes, news)
        ├── FileProcessor → reads uploaded document
        └── direct answer
  └── Amazon Polly        → text to speech (matches user's language)
  └── Audio streamed back to phone via LiveKit
```

---

## Document Processing

When a user uploads a file, it arrives over the LiveKit data channel and is routed by `FileProcessor`:

| File Type | Processor |
|---|---|
| PDF | Sarvam Document Intelligence |
| Image (JPG, PNG) | Sarvam Document Intelligence |
| Excel (.xlsx) | openpyxl |
| CSV | stdlib csv |
| Unknown | Sarvam Document Intelligence (best-effort) |

Extracted text is injected into Claude's context, and Arrow reads it out in plain spoken language.

---

## Deploying to EC2 (t3.medium)

### 1. Install dependencies

```bash
sudo apt update && sudo apt install python3-pip python3-venv git -y
git clone https://github.com/19Doors/ArrowBackend
cd ArrowBackend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment file

```bash
sudo nano /etc/agents.env
# Paste all your environment variables
sudo chmod 600 /etc/agents.env
```

### 3. Create systemd service

```bash
sudo nano /etc/systemd/system/agents.service
```

```ini
[Unit]
Description=Arrow Agent Worker
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ArrowBackend
EnvironmentFile=/etc/agents.env
ExecStart=/home/ubuntu/ArrowBackend/venv/bin/python agents.py start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable agents
sudo systemctl start agents
```

### 4. Check logs

```bash
sudo journalctl -u agents -f
```

---

## CI/CD (GitHub Actions)

Every push to `main` auto-deploys to EC2:

```yaml
# .github/workflows/deploy.yml
name: Deploy to EC2

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: SSH and deploy
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd ArrowBackend
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            sudo systemctl restart agents
```

Add `EC2_HOST` and `EC2_SSH_KEY` to your GitHub repo's Secrets.

---

## LiveKit EC2 Setup (t3.small)

The LiveKit server runs on a separate EC2 instance. Required open ports:

| Port | Protocol | Purpose |
|---|---|---|
| 80 | TCP | HTTP (Caddy redirect) |
| 443 | TCP | HTTPS / WSS (Caddy TLS) |
| 7881 | TCP | LiveKit WebSocket |
| 3478 | UDP | TURN Server |
| 50000–60000 | UDP | WebRTC media range |

---

## AWS IAM Policy

Scope your IAM user to only what Arrow needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "transcribe:StartStreamTranscription",
        "polly:SynthesizeSpeech",
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## License

MIT
