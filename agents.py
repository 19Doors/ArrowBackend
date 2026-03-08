"""
Grameen AI — Rural Community Voice Agent
─────────────────────────────────────────
STT  : Sarvam saaras:v3   (auto-detect language)
LLM  : GLM-4.7 via Nano   (swappable — see commented AWS block)
TTS  : Sarvam bulbul:v3   (language-matched from STT)
Files: LiveKit Byte Streams on topic "document-upload"
       images → Sarvam Vision OCR
       PDFs   → pypdf text layer → Vision fallback
       Excel  → openpyxl
       CSV    → stdlib csv
RPC  : "reanalyze-document" — re-trigger spoken analysis with custom focus
Search: Exa MCP
"""

from datetime import date
import asyncio
import base64
import io
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, mcp
from livekit.agents.voice import room_io
from livekit.plugins import silero, sarvam, openai
from livekit.plugins import aws   # ← uncomment to switch LLM

load_dotenv(".env")
logger = logging.getLogger("grameen-ai")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_API_KEY    = os.environ["SARVAM_API_KEY"]
EXA_API_KEY       = os.environ["EXA_API_KEY"]
SARVAM_VISION_URL = "https://api.sarvam.ai/v1/vision"   # verify against latest docs

MAX_IMAGE_PX  = 1024
MAX_PDF_PAGES = 6

BYTE_STREAM_TOPIC = "document-upload"
RPC_REANALYZE     = "reanalyze-document"

# ─────────────────────────────────────────────────────────────────────────────
# Language map
# ─────────────────────────────────────────────────────────────────────────────

LANG_MAP: dict[str, str] = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "bn": "bn-IN",
    "pa": "pa-IN", "or": "or-IN", "en": "en-IN", "unknown": "en-IN",
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a helpful community assistant for people in rural India. Today is {date.today()}.

TOOLS:
- You have access to a web search tool powered by Exa. Use it whenever you need current information such as:
  - Today's mandi prices or crop rates
  - Latest government scheme updates, deadlines, or new notifications
  - Current weather or pest outbreak alerts
  - Any fact you are not certain about
- Always search before answering questions about prices, laws, or recent events.
- After searching, summarise the result in simple spoken language — never read out URLs or raw text.

VOICE RULES — follow these strictly:
- Speak in short, clear sentences. Never use bullet points, asterisks, or markdown.
- Never say things like "Here are three points" or use numbered lists out loud. Weave information naturally into speech.
- Pause naturally. Use commas and full stops to create breath points.
- Keep every response under 60 words unless the user specifically asks for more detail.

DOCUMENT RULES — when extracted document text is given to you:
- First confirm what the document appears to be in one short sentence.
- Read out the most important information clearly, in natural spoken order.
- If it is a government form, name the scheme or department it belongs to.
- If fields are blank or illegible, say so plainly.
- Never read out long codes, reference numbers, or file paths verbatim — summarise them.

LANGUAGE:
- Use simple, everyday words. Avoid jargon. If you must use a technical term, explain it immediately.

YOUR JOB:
- Agriculture: crop cycles, pest control, soil care, basic animal health.
- Government schemes: PM-KISAN, MGNREGA, subsidies, pensions. Exact documents and where to submit.
- Basic health: hygiene, first aid, prevention. Always refer to PHC, ASHA worker, or local doctor for illness.
- Money and education: safe banking, microloans, mobile payments, children's learning resources.

HARD LIMITS:
- Never diagnose illness or suggest medicine.
- Never advise on risky investments.
- If you do not know a real-time price, law, or weather detail, search first, then answer honestly if nothing is found.
- For step-by-step processes, speak each step as a natural sentence in sequence.

"""

# ─────────────────────────────────────────────────────────────────────────────
# File processor
# ─────────────────────────────────────────────────────────────────────────────

class FileProcessor:
    """
    Routes raw bytes → plain text for the LLM.

    image/*                      → Sarvam Vision API
    application/pdf              → pypdf text layer; Vision OCR fallback
    application/vnd.*excel* .xls → openpyxl
    text/csv                     → stdlib csv
    anything else                → Vision API best-effort
    """

    async def process(self, data: bytes, mime_type: str, filename: str) -> str:
        mime = mime_type.lower()
        try:
            if mime.startswith("image/"):
                return await self._process_image(data, mime)
            elif mime == "application/pdf":
                return await self._process_pdf(data)
            elif "spreadsheet" in mime or "excel" in mime or mime == "application/vnd.ms-excel":
                return await asyncio.to_thread(self._process_excel, data)
            elif mime in ("text/csv", "application/csv"):
                return await asyncio.to_thread(self._process_csv, data)
            else:
                logger.warning("Unknown mime %s for '%s' — trying Vision", mime, filename)
                return await self._process_image(data, "image/jpeg")
        except Exception as exc:
            logger.exception("FileProcessor error for '%s'", filename)
            return f"[Error processing '{filename}': {exc}]"

    # ── Image ──────────────────────────────────────────────────────────────

    async def _process_image(self, data: bytes, mime: str) -> str:
        b64 = await asyncio.to_thread(self._resize_encode, data, mime)
        return await self._vision_ocr(b64, mime)

    @staticmethod
    def _resize_encode(data: bytes, mime: str) -> str:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG" if "jpeg" in mime or "jpg" in mime else "PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            return base64.b64encode(data).decode()

    @staticmethod
    async def _vision_ocr(b64: str, mime: str) -> str:
        payload = {
            "model": "sarvam-vision",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": (
                        "Extract ALL text from this document or image exactly as written. "
                        "If it is a form, list every field name and its filled value. "
                        "If it is a photograph of crops, fields, or animals, describe what you see. "
                        "Output plain text only — no markdown, no bullet points."
                    )},
                ],
            }],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                SARVAM_VISION_URL,
                json=payload,
                headers={"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # ── PDF ────────────────────────────────────────────────────────────────

    async def _process_pdf(self, data: bytes) -> str:
        text = await asyncio.to_thread(self._pdf_text_layer, data)
        if text and len(text.strip()) > 50:
            return f"[PDF text]\n{text}"

        logger.info("PDF has no text layer — using Vision OCR")
        pages = await asyncio.to_thread(self._pdf_to_images, data)
        if not pages:
            return "[Scanned PDF: pdf2image/poppler not installed]"

        results = []
        for i, (img, img_mime) in enumerate(pages):
            b64 = base64.b64encode(img).decode()
            results.append(f"--- Page {i+1} ---\n{await self._vision_ocr(b64, img_mime)}")
        return "\n\n".join(results)

    @staticmethod
    def _pdf_text_layer(data: bytes) -> str:
        try:
            from pypdf import PdfReader
            return "\n\n".join(
                p.extract_text() or ""
                for p in PdfReader(io.BytesIO(data)).pages[:MAX_PDF_PAGES]
            )
        except ImportError:
            logger.warning("pypdf not installed")
            return ""

    @staticmethod
    def _pdf_to_images(data: bytes) -> list[tuple[bytes, str]]:
        try:
            from pdf2image import convert_from_bytes
            out = []
            for page in convert_from_bytes(data, dpi=150, first_page=1, last_page=MAX_PDF_PAGES):
                buf = io.BytesIO()
                page.save(buf, format="JPEG", quality=85)
                out.append((buf.getvalue(), "image/jpeg"))
            return out
        except ImportError:
            logger.warning("pdf2image not installed")
            return []

    # ── Excel ──────────────────────────────────────────────────────────────

    @staticmethod
    def _process_excel(data: bytes) -> str:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            parts = []
            for name in wb.sheetnames:
                rows = [
                    " | ".join(str(c) if c is not None else "" for c in row)
                    for row in wb[name].iter_rows(values_only=True)
                    if any(c is not None for c in row)
                ]
                if rows:
                    parts.append(f"Sheet: {name}\n" + "\n".join(rows))
            return "\n\n".join(parts) or "[Excel is empty]"
        except ImportError:
            return "[openpyxl not installed]"

    # ── CSV ────────────────────────────────────────────────────────────────

    @staticmethod
    def _process_csv(data: bytes) -> str:
        import csv
        rows = [
            " | ".join(row)
            for row in csv.reader(io.StringIO(data.decode("utf-8", errors="replace")))
            if any(row)
        ]
        return "\n".join(rows) or "[CSV is empty]"


# ─────────────────────────────────────────────────────────────────────────────
# Byte stream handler
# ─────────────────────────────────────────────────────────────────────────────
# Follows the LiveKit Python pattern exactly:
#   - async_handle reads the stream and does work
#   - sync handle_byte_stream wraps it in a task (required by the SDK)
#   - tasks stored in a set to prevent GC before completion
# ─────────────────────────────────────────────────────────────────────────────

_processor = FileProcessor()
_active_tasks: set[asyncio.Task] = set()


def register_byte_stream_handler(room: rtc.Room, session: AgentSession) -> None:

    async def _handle(reader, participant_identity: str) -> None:
        info      = reader.info
        # reader.info properties: name, mime_type, topic, timestamp, id, size
        filename  = info.name      if hasattr(info, "name")      else info.get("name", "document")
        mime_type = info.mime_type if hasattr(info, "mime_type") else info.get("mime_type", "application/octet-stream")

        logger.info("Receiving '%s' (%s) from %s", filename, mime_type, participant_identity)

        # Acknowledge receipt immediately — don't leave the user in silence
        await session.generate_reply(
                user_input=f"Tell the user in one short sentence that you received their document and are reading it. document filename: {filename}"
        )

        # Collect all chunks (mirrors the LiveKit Python example pattern)
        chunks: list[bytes] = []
        async for chunk in reader:
            chunks.append(chunk)
        file_bytes = b"".join(chunks)

        logger.info("Received %d bytes for '%s'", len(file_bytes), filename)

        extracted = await _processor.process(file_bytes, mime_type, filename)

        # ── Persist document in chat history ─────────────────────────────
        # chat_ctx.append() adds this to the LLM's conversation history
        # permanently, so follow-up questions ("what was the name on that
        # form?") work in all future turns — not just the immediate reply.
        chat_ctx = session.current_agent.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=f"[Document uploaded: {filename}]\n\n{extracted}")
        await session.current_agent.update_chat_ctx(chat_ctx)
        # session.chat_ctx.append(
        #     role="user",
        #     text=(
        #         f"[I uploaded a document: '{filename}']\n\n"
        #         f"{extracted}"
        #     ),
        # )

        # generate_reply sees the full context including the doc above.
        await session.generate_reply(
                instructions="""
                Acknowledge that you got the document/s!
                """
        )

    def _sync_handle(reader, participant_identity: str) -> None:
        task = asyncio.create_task(_handle(reader, participant_identity))
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)

    room.register_byte_stream_handler(BYTE_STREAM_TOPIC, _sync_handle)
    logger.info("Byte stream handler ready on topic '%s'", BYTE_STREAM_TOPIC)


# ─────────────────────────────────────────────────────────────────────────────
# RPC handler
# ─────────────────────────────────────────────────────────────────────────────
# Frontend calls:
#   room.localParticipant.performRpc({
#     destinationIdentity: '<agent-identity>',
#     method: 'reanalyze-document',
#     payload: JSON.stringify({ instruction: 'Focus only on expiry dates.' })
#   })

def register_rpc_methods(room: rtc.Room, session: AgentSession) -> None:

    @room.local_participant.register_rpc_method(RPC_REANALYZE)
    async def handle_reanalyze(data) -> str:
        try:
            body        = json.loads(data.payload or "{}")
            instruction = body.get("instruction", "Re-summarise the last document sent.")
        except json.JSONDecodeError:
            instruction = data.payload or "Re-summarise the last document sent."

        logger.info("RPC reanalyze from %s: %s", data.caller_identity, instruction)
        await session.generate_reply(instructions=instruction)
        return "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


# ─────────────────────────────────────────────────────────────────────────────
# Session handler
# ─────────────────────────────────────────────────────────────────────────────

server = AgentServer()


@server.rtc_session()
async def session_handler(ctx: agents.JobContext):

    # ── STT ─────────────────────────────────────────────────────────────────
    stt = sarvam.STT(
        language="unknown",
        model="saaras:v3",
        mode="transcribe",
        api_key=SARVAM_API_KEY,
    )

    # ── Language-aware TTS ───────────────────────────────────────────────────
    detected_lang: list[str] = ["unknown"]

    def on_transcript(transcript) -> None:
        detected_lang[0] = getattr(transcript, "language", "unknown") or "unknown"

    stt.on("transcript", on_transcript)

    def make_tts() -> sarvam.TTS:
        return sarvam.TTS(
            target_language_code=LANG_MAP.get(detected_lang[0], "en-IN"),
            model="bulbul:v3",
            speaker="shubh",
            api_key=SARVAM_API_KEY,
            pace=1.15,
        )

    # ── LLM ─────────────────────────────────────────────────────────────────
    # llm = openai.LLM(
    #     model="zai-org/glm-4.7-original:thinking",
    #     api_key=os.environ["NANO_API_KEY"],
    #     base_url=os.environ["NANO_BASE_URL"],
    #     temperature=0.4,
    # )
    # Swap to AWS Bedrock anytime:
    llm = aws.LLM(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # model="openai.gpt-oss-safeguard-120b",
        temperature=0.4,
    )

    # ── Web search ───────────────────────────────────────────────────────────
    exa_mcp = mcp.MCPServerStdio(
        command="bunx",
        args=["-y", "exa-mcp-server"],
        env={"EXA_API_KEY": EXA_API_KEY},
        client_session_timeout_seconds=120,
    )

    # ── Session ──────────────────────────────────────────────────────────────
    session = AgentSession(
        llm=llm,
        stt=stt,
        tts=make_tts(),
        vad=silero.VAD.load(),
        mcp_servers=[exa_mcp],
        use_tts_aligned_transcript=True
    )

    # Byte stream handler can be registered before connect — it just
    # attaches a listener and doesn't touch local_participant.
    register_byte_stream_handler(ctx.room, session)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=room_io.RoomInputOptions(close_on_disconnect=True),
    )

    # RPC must be registered AFTER session.start() because it needs
    # room.local_participant, which only exists once the room is connected.
    register_rpc_methods(ctx.room, session)

    if ctx.room.remote_participants:
        await session.generate_reply(
            instructions="Greet the user warmly in one short sentence. Do not mention capabilities yet."
        )


if __name__ == "__main__":
    agents.cli.run_app(server)
