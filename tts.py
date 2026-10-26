import io
import os
import edge_tts

TTS_VOICE = os.getenv("TTS_VOICE", "pt-BR-AntonioNeural")

async def synthesize_speech(text: str, voice: str = None) -> bytes:
    """
    Sintetiza o texto em áudio MP3 de alta fidelidade usando edge-tts.
    Não consome GPU e tem latência quase instantânea.
    """
    selected_voice = voice or TTS_VOICE
    communicate = edge_tts.Communicate(text=text, voice=selected_voice, rate="+5%")
    
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
            
    mp3_buffer.seek(0)
    return mp3_buffer.read()
