import os
import json
import re
from typing import List, Dict, Any, Tuple
from groq import AsyncGroq
from tools import TOOL_DEFINITIONS, execute_tool

SYSTEM_PROMPT = """Você é o Charles, uma inteligência artificial assistente pessoal e copiloto operacional inspirada no J.A.R.V.I.S. do Homem de Ferro.

Sua personalidade e tom:
- Cordial, refinado, britânico, altamente técnico, perspicaz e leal.
- Chame o usuário com deferência respeitosa ("Senhor" ou pelo nome quando apropriado).
- Respostas curtas, elegantes e práticas: suas respostas serão sintetizadas em áudio (voz) no smartphone ou computador.
- Nunca use asteriscos, markdown pesado, tabelas ou listas numéricas longas no texto de fala.
- Se executar uma ação (como abrir WhatsApp ou buscar dados), reporte com tranquilidade executiva (ex: "Certamente, senhor. Iniciando a comunicação no WhatsApp.").
- Responda sempre no idioma em que o senhor se comunicar (preferencialmente Português do Brasil)."""

# Modelos recomendados da Groq que suportam Tool Calling comprovadamente
SUPPORTED_TOOL_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "mixtral-8x7b-32768"
]

class CharlesOrchestrator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY não foi encontrada nas variáveis de ambiente.")
        self.client = AsyncGroq(api_key=self.api_key)
        self.default_model = "llama-3.1-8b-instant"

    async def list_available_models(self) -> List[str]:
        """Consulta modelos da Groq e filtra estritamente os compatíveis com Tool Calling."""
        try:
            models_response = await self.client.models.list()
            active_ids = {m.id for m in models_response.data}
            
            # Intersecta com os modelos que suportam Function Calling
            filtered = [m for m in SUPPORTED_TOOL_MODELS if m in active_ids]
            
            # Adiciona outros modelos llama se existirem
            for mid in active_ids:
                if mid.startswith("llama-3") and mid not in filtered and "guard" not in mid:
                    filtered.append(mid)

            return filtered if filtered else ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
        except Exception:
            return ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Envia o áudio gravado para a Groq Whisper-large-v3-turbo."""
        try:
            transcription = await self.client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language="pt",
                temperature=0.0
            )
            return transcription.text.strip()
        except Exception:
            transcription = await self.client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3",
                response_format="verbose_json",
                language="pt",
                temperature=0.0
            )
            return transcription.text.strip()

    async def chat(self, messages: List[Dict[str, Any]], device_telemetry: Dict[str, Any] = None, selected_model: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        mobile_actions = []
        target_model = selected_model if (selected_model and selected_model in SUPPORTED_TOOL_MODELS) else self.default_model
        
        telemetry_info = ""
        if device_telemetry:
            telemetry_info = f"\n[Telemetria do Dispositivo]: Bateria: {device_telemetry.get('battery', 'N/D')}, Conexão: {device_telemetry.get('connection', 'N/D')}."

        system_message = SYSTEM_PROMPT + telemetry_info
        payload_messages = [{"role": "system", "content": system_message}] + messages

        response = await self.client.chat.completions.create(
            model=target_model,
            messages=payload_messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=500
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            payload_messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except Exception:
                    func_args = {}

                tool_result = execute_tool(func_name, func_args)

                if func_name == "create_mobile_action" and isinstance(tool_result, dict):
                    mobile_actions.append(tool_result)

                payload_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            final_response = await self.client.chat.completions.create(
                model=target_model,
                messages=payload_messages,
                temperature=0.3,
                max_tokens=350
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = response_message.content

        clean_text = self._sanitize_for_tts(final_text)
        return clean_text, mobile_actions

    def _sanitize_for_tts(self, text: str) -> str:
        text = re.sub(r'[*#_`]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        return text.strip()
