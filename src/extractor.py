"""
Extração e limpeza de texto de PDFs institucionais.
Preserva estrutura: títulos, parágrafos e tabelas convertidas para Markdown.
"""

import re
import pdfplumber
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str      # nome do arquivo
    page: int
    chunk_id: str    # source:page:index


def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-záéíóúãõâêîôûàüç])- ([a-záéíóúãõâêîôûàüç])', r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    def cell(v):
        return str(v).replace('\n', ' ').strip() if v else ""
    sep = " | ".join(["---"] * len(header))
    lines = ["| " + " | ".join(cell(c) for c in header) + " |",
             "| " + sep + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def extract_pages(pdf_path: Path) -> list[dict]:
    """Retorna lista de {page, text} com texto limpo por página."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extrai tabelas primeiro para não duplicar conteúdo
            tables = page.extract_tables()
            table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

            # Remove regiões de tabela do texto principal
            words = page.extract_words()
            table_texts: list[str] = []
            for table in tables:
                md = _table_to_markdown(table)
                if md:
                    table_texts.append(md)

            # Texto fora das tabelas
            if table_bboxes:
                # filtra palavras dentro de bboxes de tabelas
                def in_table(w):
                    x0, top, x1, bottom = w['x0'], w['top'], w['x1'], w['bottom']
                    for bx0, btop, bx1, bbottom in table_bboxes:
                        if x0 >= bx0 and x1 <= bx1 and top >= btop and bottom <= bbottom:
                            return True
                    return False
                words = [w for w in words if not in_table(w)]

            body = " ".join(w['text'] for w in words)
            body = _clean_text(body)

            parts = [body] + table_texts
            full = "\n\n".join(p for p in parts if p)
            if full.strip():
                pages.append({"page": i, "text": full})
    return pages


def chunk_pages(pages: list[dict], source: str,
                chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """
    Divide o texto por páginas em chunks de ~chunk_size tokens (aprox. chars/4).
    Overlap garante continuidade entre chunks.
    """
    chunks: list[Chunk] = []
    for p in pages:
        text = p["text"]
        page_num = p["page"]
        words = text.split()
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if len(chunk_text.strip()) > 50:  # ignora chunks mínimos
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    page=page_num,
                    chunk_id=f"{source}:p{page_num}:c{idx}",
                ))
                idx += 1
            if end == len(words):
                break
            start = end - overlap
    return chunks


def extract_document(pdf_path: Path, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    source = pdf_path.name
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, source, chunk_size, overlap)
