"""
Avaliação do sistema RAG: testa perguntas reais e analisa erros.
Gera relatório com métricas de recuperação e qualidade das respostas.
"""

import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.assistant import Assistant
from src.vectorstore import VectorStore

console = Console()

# Perguntas reais de alunos com respostas esperadas (ground truth parcial)
TEST_QUESTIONS = [
    {
        "question": "Qual é a carga horária total do curso de Ciência da Computação?",
        "keywords": ["carga horária", "hora", "crédito"],
        "doc_hint": "PPC_CComputacao",
    },
    {
        "question": "Como funciona o estágio supervisionado obrigatório?",
        "keywords": ["estágio", "supervisor", "carga horária", "obrigatório"],
        "doc_hint": "manual_de_estagio",
    },
    {
        "question": "Quais são os prazos para matrícula no segundo semestre de 2026?",
        "keywords": ["matrícula", "prazo", "2026"],
        "doc_hint": "Edital_de_Matricula",
    },
    {
        "question": "Quais atividades complementares são aceitas no curso de Pedagogia?",
        "keywords": ["atividade complementar", "pedagogia", "hora"],
        "doc_hint": "PEDAGOGIA",
    },
    {
        "question": "Qual é o prazo mínimo e máximo para integralização do curso?",
        "keywords": ["prazo", "integralização", "semestre", "ano"],
        "doc_hint": "PPC",
    },
    {
        "question": "Como solicitar aproveitamento de disciplinas cursadas em outra instituição?",
        "keywords": ["aproveitamento", "disciplina", "equivalência"],
        "doc_hint": "manual",
    },
    {
        "question": "Quais são as normas para o TCC?",
        "keywords": ["TCC", "trabalho de conclusão", "monografia"],
        "doc_hint": "PPC",
    },
    {
        "question": "Qual é o calendário acadêmico do segundo semestre de 2026?",
        "keywords": ["calendário", "2026", "semestre"],
        "doc_hint": "Calendario",
    },
]


def _check_keywords(answer: str, keywords: list[str]) -> float:
    """Proporção de keywords encontradas na resposta (recall superficial)."""
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 2) if keywords else 0.0


def _check_source_match(hits: list[dict], doc_hint: str) -> bool:
    """Verifica se o documento esperado aparece nos chunks recuperados."""
    return any(doc_hint.lower() in h["source"].lower() for h in hits)


def run_evaluation(assistant: Assistant, output_path: str = "data/evaluation.json") -> None:
    results = []
    console.print("\n[bold]Iniciando avaliação com perguntas reais...[/bold]\n")

    for test in TEST_QUESTIONS:
        q = test["question"]
        console.print(f"[cyan]Q:[/cyan] {q}")
        try:
            resp = assistant.ask(q)
            kw_score = _check_keywords(resp["answer"], test["keywords"])
            src_match = _check_source_match(resp["hits"], test["doc_hint"])

            result = {
                "question": q,
                "answer": resp["answer"],
                "keyword_recall": kw_score,
                "source_matched": src_match,
                "sources_retrieved": [h["source"] for h in resp["hits"]],
                "top_score": resp["hits"][0]["score"] if resp["hits"] else 0,
                "usage": resp.get("usage", {}),
                "error": None,
            }
            status = "[green]✓[/green]" if kw_score >= 0.5 and src_match else "[yellow]~[/yellow]"
            console.print(f"  {status} keyword_recall={kw_score} | source_match={src_match}\n")
        except Exception as e:
            result = {"question": q, "error": str(e)}
            console.print(f"  [red]ERRO:[/red] {e}\n")

        results.append(result)

    # Métricas agregadas
    valid = [r for r in results if not r.get("error")]
    avg_kw = round(sum(r["keyword_recall"] for r in valid) / len(valid), 2) if valid else 0
    src_hits = sum(1 for r in valid if r["source_matched"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_keyword_recall": avg_kw,
        "source_match_rate": round(src_hits / len(valid), 2) if valid else 0,
        "results": results,
    }

    # Tabela resumo
    table = Table(title="Resumo da Avaliação")
    table.add_column("Métrica", style="bold")
    table.add_column("Valor")
    table.add_row("Total de perguntas", str(summary["total_questions"]))
    table.add_row("Erros", str(summary["errors"]))
    table.add_row("Keyword recall médio", str(avg_kw))
    table.add_row("Taxa de fonte correta", str(summary["source_match_rate"]))
    console.print(table)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    console.print(f"\n[bold green]Relatório salvo em {output_path}[/bold green]")
