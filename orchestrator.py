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

class CharlesOrchestrator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY não foi encontrada nas variáveis de ambiente.")
        self.client = AsyncGroq(api_key=self.api_key)

    async def list_available_models(self) -> List[str]:
        """Consulta os modelos ativos reais da conta da Groq."""
        try:
            models_response = await self.client.models.list()
            chat_models = []
            for m in models_response.data:
                mid = m.id
                # Ignora whisper e guard
                if not mid.startswith("whisper") and not "guard" in mid and not "safeguard" in mid:
                    chat_models.append(mid)
            return sorted(chat_models) if chat_models else ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
        except Exception:
            return ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Envia o áudio gravado para a Groq Whisper com fallback automático."""
        whisper_models = ["whisper-large-v3-turbo", "whisper-large-v3"]
        last_err = None
        for wm in whisper_models:
            try:
                transcription = await self.client.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model=wm,
                    response_format="verbose_json",
                    language="pt",
                    temperature=0.0
                )
                return transcription.text.strip()
            except Exception as e:
                last_err = e
                continue
        raise last_err or Exception("Falha na transcrição de áudio.")

    async def chat(self, messages: List[Dict[str, Any]], device_telemetry: Dict[str, Any] = None, selected_model: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executa a conversa com resiliência: se o modelo selecionado falhar,
        tenta outros modelos ativos da conta automaticamente.
        """
        available_models = await self.list_available_models()
        candidate_models = []
        if selected_model and selected_model in available_models:
            candidate_models.append(selected_model)
        for m in available_models:
            if m not in candidate_models:
                candidate_models.append(m)

        mobile_actions = []
        telemetry_info = ""
        if device_telemetry:
            telemetry_info = f"\n[Telemetria do Dispositivo]: Bateria: {device_telemetry.get('battery', 'N/D')}, Conexão: {device_telemetry.get('connection', 'N/D')}."

        system_message = SYSTEM_PROMPT + telemetry_info
        payload_messages = [{"role": "system", "content": system_message}] + messages

        last_error = None
        for current_model in candidate_models:
            try:
                # Tenta chamada com Tools
                try:
                    response = await self.client.chat.completions.create(
                        model=current_model,
                        messages=payload_messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        temperature=0.3,
                        max_tokens=500
                    )
                except Exception as tool_err:
                    # Se o modelo não suportar tools, tenta sem tools
                    if "tool" in str(tool_err).lower() or "not supported" in str(tool_err).lower():
                        response = await self.client.chat.completions.create(
                            model=current_model,
                            messages=payload_messages,
                            temperature=0.3,
                            max_tokens=500
                        )
                    else:
                        raise tool_err

                response_message = response.choices[0].message

                if getattr(response_message, 'tool_calls', None):
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
                        model=current_model,
                        messages=payload_messages,
                        temperature=0.3,
                        max_tokens=350
                    )
                    final_text = final_response.choices[0].message.content
                else:
                    final_text = response_message.content

                clean_text = self._sanitize_for_tts(final_text)
                return clean_text, mobile_actions

            except Exception as e:
                last_error = e
                continue

        raise last_error or Exception("Nenhum modelo da conta Groq respondeu com sucesso.")

    def _sanitize_for_tts(self, text: str) -> str:
        text = re.sub(r'[*#_`]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        return text.strip()
