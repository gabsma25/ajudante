"""
Integração com a API da Anthropic (Claude).
Injeta chunks recuperados no prompt e gera resposta fundamentada nos documentos.
"""

import os
import anthropic
from dotenv import load_dotenv
from src.vectorstore import VectorStore

load_dotenv()


def _build_client() -> anthropic.Anthropic:
    """Cria cliente Anthropic usando API key ou token OAuth de sessão."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file:
        with open(token_file) as f:
            return anthropic.Anthropic(auth_token=f.read().strip())
    raise RuntimeError(
        "Defina ANTHROPIC_API_KEY no arquivo .env ou execute em sessão Claude Code."
    )


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
                 model: str = "claude-haiku-4-5-20251001",
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

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        sources = list({f"{h['source']} (p. {h['page']})" for h in hits})

        return {
            "answer": answer,
            "sources": sources,
            "hits": hits,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
