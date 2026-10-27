import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from orchestrator import CharlesOrchestrator
from tts import synthesize_speech

load_dotenv()

app = FastAPI(title="C.H.A.R.L.E.S. - Personal AI Operating System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

orchestrator = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada no servidor.")
        orchestrator = CharlesOrchestrator(api_key=api_key)
    return orchestrator

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "online", "system": "C.H.A.R.L.E.S. Operational Core"}

@app.get("/api/models")
async def get_models():
    try:
        orch = get_orchestrator()
        models = await orch.list_available_models()
        return {"models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]})

@app.post("/api/interact-voice")
async def interact_voice(
    audio: UploadFile = File(...),
    history: str = Form("[]"),
    telemetry: str = Form("{}"),
    model: str = Form("openai/gpt-oss-120b")
):
    try:
        orch = get_orchestrator()
        audio_bytes = await audio.read()
        
        # 1. Transcrição de Voz via Groq Whisper
        user_text = await orch.transcribe_audio(audio_bytes, filename=audio.filename or "audio.webm")
        if not user_text:
            return JSONResponse(status_code=400, content={"error": "Nenhuma fala detectada."})

        # 2. Histórico da conversa
        try:
            conversation_history = json.loads(history)
        except Exception:
            conversation_history = []

        try:
            device_telemetry = json.loads(telemetry)
        except Exception:
            device_telemetry = {}

        conversation_history.append({"role": "user", "content": user_text})

        # 3. Processamento na IA da Groq
        reply_text, mobile_actions = await orch.chat(conversation_history, device_telemetry, selected_model=model)

        # 4. Síntese de Voz com fallback seguro
        tts_audio = await synthesize_speech(reply_text)
        audio_base64 = base64.b64encode(tts_audio).decode("utf-8") if tts_audio else None

        return {
            "transcription": user_text,
            "response_text": reply_text,
            "audio_base64": audio_base64,
            "mobile_actions": mobile_actions
        }

    except Exception as e:
        return JSONResponse(status_code=200, content={
            "transcription": "Comando recebido",
            "response_text": f"Peço desculpas, senhor: {str(e)}",
            "audio_base64": None,
            "mobile_actions": []
        })

@app.post("/api/interact-text")
async def interact_text(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "").strip()
        history = data.get("history", [])
        device_telemetry = data.get("telemetry", {})
        model = data.get("model", "openai/gpt-oss-120b")

        if not user_text:
            return JSONResponse(status_code=400, content={"error": "Texto vazio."})

        orch = get_orchestrator()
        history.append({"role": "user", "content": user_text})
        
        reply_text, mobile_actions = await orch.chat(history, device_telemetry, selected_model=model)
        
        tts_audio = await synthesize_speech(reply_text)
        audio_base64 = base64.b64encode(tts_audio).decode("utf-8") if tts_audio else None

        return {
            "response_text": reply_text,
            "audio_base64": audio_base64,
            "mobile_actions": mobile_actions
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "response_text": f"Peço desculpas, senhor: {str(e)}",
            "audio_base64": None,
            "mobile_actions": []
        })
