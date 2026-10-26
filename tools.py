"""
Conjunto de ferramentas (tools) para o Charles:
- Busca Web em Tempo Real
- Ações Rápidas de Celular (WhatsApp, Tel, E-mail, Mapas)
- Relatório Matinal Executivo
- Horário e Data Exatos
"""

import json
from datetime import datetime
from urllib.parse import quote_plus
from duckduckgo_search import DDGS

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Busca informações atualizadas na web como notícias, clima, esportes, preços ou fatos recentes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Termo de busca objetivo e claro"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_mobile_action",
            "description": "Gera uma ação direta executável no smartphone (ex: abrir conversa no WhatsApp, discar telefone, abrir localização no mapa, enviar e-mail).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["whatsapp", "phone_call", "map_location", "send_email", "open_link"],
                        "description": "Tipo de ação mobile desejada"
                    },
                    "target": {
                        "type": "string",
                        "description": "Destino: número de telefone com DDD (WhatsApp/Call), endereço/local (mapa), e-mail ou URL completa"
                    },
                    "message_or_subject": {
                        "type": "string",
                        "description": "Mensagem pré-preenchida (para WhatsApp/e-mail) ou termo de busca"
                    }
                },
                "required": ["action_type", "target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_executive_briefing",
            "description": "Fornece o resumo executivo matinal ou geral do dia: data, hora, previsão e principais acontecimentos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Cidade do usuário para previsão do tempo e notícias locais (opcional, padrão: Brasil)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time_and_date",
            "description": "Retorna o dia, mês, ano, dia da semana e horário atual com exatidão.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def tool_web_search(query: str) -> str:
    """Executa busca na web via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "Nenhum resultado relevante encontrado."
            
            snippets = []
            for r in results:
                snippets.append(f"Título: {r.get('title')}\nConteúdo: {r.get('body')}\nFonte: {r.get('href')}")
            return "\n\n---\n\n".join(snippets)
    except Exception as e:
        return f"Erro na pesquisa web: {str(e)}"

def tool_create_mobile_action(action_type: str, target: str, message_or_subject: str = "") -> dict:
    """Cria links diretos (Deep Links) que o celular executa nativamente."""
    url = ""
    description = ""
    
    if action_type == "whatsapp":
        clean_num = "".join(ch for ch in target if ch.isdigit())
        msg = quote_plus(message_or_subject)
        url = f"https://wa.me/{clean_num}?text={msg}" if msg else f"https://wa.me/{clean_num}"
        description = f"Abrir WhatsApp para {target}"
        
    elif action_type == "phone_call":
        clean_num = "".join(ch for ch in target if ch.isdigit() or ch == "+")
        url = f"tel:{clean_num}"
        description = f"Discar para {clean_num}"
        
    elif action_type == "map_location":
        encoded_loc = quote_plus(target)
        url = f"https://www.google.com/maps/search/?api=1&query={encoded_loc}"
        description = f"Ver no mapa: {target}"
        
    elif action_type == "send_email":
        sub = quote_plus(message_or_subject)
        url = f"mailto:{target}?subject={sub}"
        description = f"Enviar e-mail para {target}"
        
    elif action_type == "open_link":
        url = target if target.startswith("http") else f"https://{target}"
        description = f"Abrir link {url}"
        
    return {
        "action_type": action_type,
        "url": url,
        "description": description
    }

def tool_get_executive_briefing(city: str = "") -> str:
    """Monta os dados para o briefing executivo estilo Jarvis."""
    now = datetime.now()
    dias = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    dia_semana = dias[now.weekday()]
    
    city_query = f"clima {city}" if city else "clima Brasil hoje"
    news_query = "noticias principais hoje Brasil"
    
    weather_info = tool_web_search(city_query)
    news_info = tool_web_search(news_query)
    
    return f"""Data e Hora: {dia_semana}, {now.strftime('%d/%m/%Y')} às {now.strftime('%H:%M')}.
Previsão/Clima:
{weather_info}

Principais acontecimentos:
{news_info}"""

def tool_get_current_time_and_date() -> str:
    """Retorna data e hora formatadas."""
    now = datetime.now()
    dias = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    dia_semana = dias[now.weekday()]
    return f"Data atual: {dia_semana}, {now.strftime('%d/%m/%Y')}. Horário atual: {now.strftime('%H:%M:%S')}."

def execute_tool(tool_name: str, arguments: dict):
    if tool_name == "web_search":
        return tool_web_search(arguments.get("query", ""))
    elif tool_name == "create_mobile_action":
        return tool_create_mobile_action(
            action_type=arguments.get("action_type", ""),
            target=arguments.get("target", ""),
            message_or_subject=arguments.get("message_or_subject", "")
        )
    elif tool_name == "get_executive_briefing":
        return tool_get_executive_briefing(arguments.get("city", ""))
    elif tool_name == "get_current_time_and_date":
        return tool_get_current_time_and_date()
    else:
        return f"Ferramenta {tool_name} não implementada."
