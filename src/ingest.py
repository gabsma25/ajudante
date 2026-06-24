"""
Pipeline de ingestão: lê todos os PDFs da pasta docs/, extrai, divide e indexa.
"""

from pathlib import Path
from rich.console import Console
from rich.progress import track
from src.extractor import extract_document
from src.vectorstore import VectorStore

console = Console()


def ingest_all(docs_dir: str = "docs", persist_dir: str = "data/chroma",
               chunk_size: int = 600, overlap: int = 100) -> VectorStore:
    store = VectorStore(persist_dir=persist_dir)
    pdfs = list(Path(docs_dir).glob("*.pdf"))

    if not pdfs:
        console.print(f"[red]Nenhum PDF encontrado em {docs_dir}/[/red]")
        return store

    console.print(f"[bold]Indexando {len(pdfs)} documento(s)...[/bold]")
    total_chunks = 0

    for pdf in track(pdfs, description="Processando PDFs"):
        try:
            chunks = extract_document(pdf, chunk_size=chunk_size, overlap=overlap)
            store.add_chunks(chunks)
            total_chunks += len(chunks)
            console.print(f"  [green]✓[/green] {pdf.name}: {len(chunks)} chunks")
        except Exception as e:
            console.print(f"  [red]✗[/red] {pdf.name}: {e}")

    console.print(f"\n[bold green]Total indexado: {store.count()} chunks no banco vetorial.[/bold green]")
    return store
