# C.H.A.R.L.E.S. — Sistema Operacional de IA Pessoal

Assistente de voz e copiloto pessoal autônomo inspirado no J.A.R.V.I.S. do Homem de Ferro, construído para máxima velocidade e responsividade no **Render**, acessível no **Smartphone** e no **PC**.

---

## ⚡ Capacidades Principais
1. **Comando de Ativação ("Charles"):** Escuta passiva no navegador que dispara a escuta ao ouvir seu nome.
2. **Telemetria do Dispositivo:** Leitura em tempo real do nível de bateria e status de carregamento do smartphone/laptop.
3. **Efeitos Sonoros Sci-Fi de HUD:** Bipes neurais sintetizados via Web Audio API ao ativar e desativar comandos.
4. **Resumo Executivo Matinal:** *"Charles, relatório matinal"* — traz dados meteorológicos, data, hora e fatos recentes.
5. **Ações Móveis Rápidas:**
   - *"Mandar mensagem no WhatsApp para [número] dizendo que chego em 10 minutos"*
   - *"Ligar para [número]"*
   - *"Ver rota até [endereço] no mapa"*
6. **Pesquisa em Tempo Real:** Conexão instantânea com notícias e web via DuckDuckGo.

---

## 🧠 Arquitetura de Velocidade
- **STT:** Groq `whisper-large-v3` (~120ms)
- **Cérebro (LLM):** Groq `llama-3.3-70b-versatile` com Function Calling (~250ms)
- **TTS:** `edge-tts` com voz neural `pt-BR-AntonioNeural` (zero custo e latência em milissegundos)

---

## 📦 Como Subir para o GitHub e Deploy no Render

### 1. Inicializar e Enviar para o seu Repositório:
No terminal na pasta do projeto:
```bash
git init
git add .
git commit -m "feat: initial commit of Charles AI OS"
git branch -M main
git remote add origin https://github.com/carlosjesusj748-blip/Charles.git
git push -u origin main
```

### 2. Deploy no Render:
1. No painel do [render.com](https://render.com), clique em **New +** -> **Web Service**.
2. Conecte o repositório `carlosjesusj748-blip/Charles`.
3. O Render utilizará o [`render.yaml`](render.yaml) automaticamente.
4. Na aba **Environment**, insira:
   - `GROQ_API_KEY`: sua chave da [Groq Console](https://console.groq.com).
5. Clique em **Deploy**.

### 3. Usar no Celular:
Abra a URL gerada pelo Render (`https://charles-xxxx.onrender.com`) no celular e selecione **"Adicionar à Tela de Início"**.
