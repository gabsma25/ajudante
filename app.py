#!/usr/bin/env python3
"""
Ajudante Universitário — CLI principal.

Uso:
  python app.py ingest              # indexa os documentos
  python app.py ask "sua pergunta"  # faz uma pergunta
  python app.py chat                # modo interativo
  python app.py evaluate            # avalia o sistema
  python app.py status              # mostra estatísticas do banco
"""

import os
import sys
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="Assistente universitário inteligente (RAG)")
console = Console()

DOCS_DIR = "docs"
PERSIST_DIR = "data/chroma"


def _get_store():
    from src.vectorstore import VectorStore
    return VectorStore(persist_dir=PERSIST_DIR)


def _get_assistant(store=None):
    from src.assistant import Assistant
    if store is None:
        store = _get_store()
    return Assistant(store)


@app.command()
def ingest(
    docs_dir: str = typer.Option(DOCS_DIR, help="Pasta com os PDFs"),
    chunk_size: int = typer.Option(600, help="Tamanho do chunk em palavras"),
    overlap: int = typer.Option(100, help="Overlap entre chunks"),
    clear: bool = typer.Option(False, "--clear", help="Limpa o banco antes de indexar"),
):
    """Extrai e indexa todos os PDFs da pasta docs/."""
    from src.ingest import ingest_all
    from src.vectorstore import VectorStore

    if clear:
        store = VectorStore(persist_dir=PERSIST_DIR)
        store.clear()
        console.print("[yellow]Banco vetorial limpo.[/yellow]")

    ingest_all(docs_dir=docs_dir, persist_dir=PERSIST_DIR,
               chunk_size=chunk_size, overlap=overlap)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta a ser respondida"),
    n_results: int = typer.Option(5, help="Número de chunks a recuperar"),
    show_sources: bool = typer.Option(True, help="Exibe trechos recuperados"),
):
    """Responde uma pergunta usando os documentos indexados."""
    assistant = _get_assistant()
    assistant._n_results = n_results

    with console.status("[bold green]Buscando e gerando resposta..."):
        result = assistant.ask(question)

    console.print(Panel(Markdown(result["answer"]), title="[bold]Resposta[/bold]", border_style="green"))

    if show_sources:
        console.print("\n[bold]Fontes utilizadas:[/bold]")
        for h in result["hits"]:
            console.print(f"  • {h['source']} — página {h['page']} (relevância: {h['score']})")

    if "usage" in result:
        u = result["usage"]
        console.print(f"\n[dim]Tokens: {u.get('input_tokens',0)} entrada / {u.get('output_tokens',0)} saída[/dim]")


@app.command()
def chat():
    """Modo interativo de perguntas e respostas."""
    assistant = _get_assistant()
    console.print(Panel(
        "[bold]Ajudante Universitário[/bold]\nDigite sua dúvida ou [bold red]sair[/bold red] para encerrar.",
        border_style="blue"
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]Você:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("sair", "exit", "quit", "q"):
            break
        if not question:
            continue

        with console.status("[bold green]Processando..."):
            result = assistant.ask(question)

        console.print(f"\n[bold green]Assistente:[/bold green]")
        console.print(Markdown(result["answer"]))
        console.print("\n[dim]Fontes: " + " | ".join(result["sources"]) + "[/dim]")

    console.print("\n[bold]Até mais![/bold]")


@app.command()
def evaluate(output: str = typer.Option("data/evaluation.json", help="Arquivo de saída")):
    """Executa avaliação com perguntas reais e salva relatório."""
    from src.evaluate import run_evaluation
    assistant = _get_assistant()
    run_evaluation(assistant, output_path=output)


@app.command()
def status():
    """Mostra estatísticas do banco vetorial."""
    store = _get_store()
    count = store.count()
    console.print(f"[bold]Banco vetorial:[/bold] {PERSIST_DIR}")
    console.print(f"[bold]Chunks indexados:[/bold] {count}")
    if count == 0:
        console.print("[yellow]Nenhum documento indexado. Execute: python app.py ingest[/yellow]")


if __name__ == "__main__":
    app()
