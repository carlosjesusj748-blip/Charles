import io
import os
import edge_tts

TTS_VOICE = os.getenv("TTS_VOICE", "pt-BR-AntonioNeural")

async def synthesize_speech(text: str, voice: str = None) -> bytes:
    """
    Tenta sintetizar via edge-tts. Se a Microsoft bloquear (como em IPs do Render),
    retorna None para que o cliente use o sintetizador de voz nativo do dispositivo.
    """
    try:
        selected_voice = voice or TTS_VOICE
        communicate = edge_tts.Communicate(text=text, voice=selected_voice, rate="+5%")
        
        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])
                
        mp3_buffer.seek(0)
        data = mp3_buffer.read()
        return data if len(data) > 0 else None
    except Exception as e:
        # Silencia erro de bloqueio da Microsoft e delega para síntese nativa do cliente
        return None
