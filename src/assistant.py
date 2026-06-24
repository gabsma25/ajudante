"""
Integração com a API do Gemini.
Injeta chunks recuperados no prompt e gera resposta fundamentada nos documentos.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types
from src.vectorstore import VectorStore

load_dotenv()


def _build_client() -> genai.Client:
    """Cria cliente Gemini usando a chave da API configurada no ambiente."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


SYSTEM_PROMPT = """Você é um assistente universitário especializado nos documentos institucionais da universidade.
Responda perguntas dos alunos com base EXCLUSIVAMENTE nos trechos de documentos fornecidos no contexto.

Regras:
- Seja preciso e objetivo; cite a fonte (nome do documento e página) ao final da resposta.
- Se a informação não estiver no contexto, diga explicitamente que não encontrou nos documentos disponíveis.
- Não invente prazos, notas ou regras que não estejam no contexto.
- Use linguagem clara e acessível para estudantes.
- Responda sempre em português."""

ANSWER_TEMPLATE = """{context}

---
Pergunta do aluno: {question}

Responda com base nos trechos acima."""


class Assistant:
    def __init__(self, vector_store: VectorStore,
                 model: str = "gemini-2.0-flash",
                 n_results: int = 5,
                 max_tokens: int = 1024):
        self._store = vector_store
        self._client = _build_client()
        self._model = model
        self._n_results = n_results
        self._max_tokens = max_tokens

    def ask(self, question: str) -> dict:
        hits = self._store.search(question, n_results=self._n_results)

        if not hits:
            return {
                "answer": "Não encontrei documentos relevantes para responder sua pergunta.",
                "sources": [],
                "hits": [],
            }

        # Monta contexto com os chunks recuperados
        context_parts = []
        for i, h in enumerate(hits, 1):
            context_parts.append(
                f"[Trecho {i} — {h['source']}, página {h['page']} | relevância: {h['score']}]\n{h['text']}"
            )
        context = "\n\n".join(context_parts)

        user_message = ANSWER_TEMPLATE.format(context=context, question=question)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self._max_tokens,
                    temperature=0.2,
                ),
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
                sources = list({f"{h['source']} (p. {h['page']})" for h in hits})
                return {
                    "answer": (
                        "Não consegui gerar a resposta porque a quota da API do Gemini foi excedida. "
                        "O índice vetorial funcionou e encontrei trechos relevantes, mas a geração do modelo "
                        "não pôde ser concluída.\n\n"
                        "Para continuar, habilite billing/quotas no projeto do Gemini ou use um modelo/API key com acesso disponível."
                    ),
                    "sources": sources,
                    "hits": hits,
                    "error": "gemini_quota_exceeded",
                }
            raise

        answer = response.text or ""
        sources = list({f"{h['source']} (p. {h['page']})" for h in hits})
        usage = getattr(response, "usage_metadata", None)

        result = {
            "answer": answer,
            "sources": sources,
            "hits": hits,
        }

        if usage:
            result["usage"] = {
                "input_tokens": getattr(usage, "prompt_token_count", 0),
                "output_tokens": getattr(usage, "candidates_token_count", 0),
            }

        return result
