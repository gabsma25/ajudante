# 🎓 Ajudante Universitário Inteligente (RAG)

Assistente capaz de responder perguntas de alunos com base em documentos
institucionais da universidade (PPCs, regulamentos, manuais de estágio,
editais e calendário acadêmico), usando **RAG** (Retrieval-Augmented
Generation) com a API do Gemini.

> **Tema 1 — Assistente Universitário Inteligente**
> Construir um assistente que responde perguntas utilizando documentos acadêmicos.

---

## 📌 Visão geral

O sistema recebe a pergunta de um aluno, busca os trechos mais relevantes
dos documentos institucionais em um banco vetorial e injeta esse contexto na
chamada ao LLM, que gera uma resposta fundamentada **exclusivamente** nos
documentos — sempre citando a fonte (documento + página).

```
Pergunta do aluno
      │
      ▼
┌─────────────────┐   embeddings   ┌──────────────────┐
│  Busca vetorial │ ◄────────────► │   ChromaDB       │
│  (similaridade) │                │  (906 chunks)    │
└────────┬────────┘                └──────────────────┘
         │ top-k chunks
         ▼
┌─────────────────┐
│  Prompt + ctx   │ ──────────────►  Gemini
└─────────────────┘                       │
                                           ▼
                                  Resposta + fontes citadas
```

---

## 🗂️ Estrutura do projeto

```
ajudante/
├── docs/                  # 11 PDFs institucionais (fonte de conhecimento)
├── src/
│   ├── extractor.py       # Extração e limpeza de texto dos PDFs + chunking
│   ├── vectorstore.py     # Banco vetorial (ChromaDB) e busca por similaridade
│   ├── assistant.py       # Integração com a API do Gemini
│   ├── ingest.py          # Pipeline de ingestão dos documentos
│   └── evaluate.py        # Avaliação automática com perguntas reais
├── data/
│   ├── chroma/            # Banco vetorial persistido (gerado por `ingest`)
│   └── evaluation.json    # Resultados da avaliação
├── app.py                 # CLI principal
├── requirements.txt
└── .env.example           # Modelo de configuração da API key
```

---

## ⚙️ Instalação

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Configure a chave da API do Gemini
cp .env.example .env
# edite o .env e coloque sua GEMINI_API_KEY
```

> Se preferir, você também pode usar `GOOGLE_API_KEY` no ambiente.

---

## 🚀 Uso

```bash
# Indexar todos os PDFs da pasta docs/ (executar uma vez)
python app.py ingest

# Fazer uma pergunta
python app.py ask "Como funciona o estágio supervisionado obrigatório?"

# Modo interativo (chat)
python app.py chat

# Rodar a avaliação automática
python app.py evaluate

# Ver estatísticas do banco vetorial
python app.py status
```

---

## 🔍 Como funciona (etapas do RAG)

### 1. Extração de texto (`extractor.py`)
- Usa **pdfplumber** para extrair texto dos PDFs.
- **Tabelas** são detectadas e convertidas para **Markdown**, e as regiões de
  tabela são removidas do texto corrido para não "sujar" o contexto.
- Limpeza: junção de palavras hifenizadas no fim de linha, normalização de
  espaços em branco.
- **Chunking** por página, com janelas de ~600 palavras e **overlap de 100**
  palavras para preservar continuidade entre trechos. Cada chunk guarda a
  origem (documento + página) nos metadados.

### 2. Embeddings + banco vetorial (`vectorstore.py`)
- Modelo de embedding: **all-MiniLM-L6-v2** (via ChromaDB), que funciona bem
  em português por ter base multilíngue.
- Armazenamento em **ChromaDB** persistente, com **similaridade de cosseno**.
- Deduplicação por `chunk_id` evita reindexar trechos já presentes.

### 3. Recuperação
- A pergunta é convertida em embedding e o banco retorna os **top-k** chunks
  mais similares (padrão: 5), com a pontuação de relevância (1 − distância).

### 4. Geração com o LLM (`assistant.py`)
- Os chunks recuperados são montados em um **contexto numerado** (com fonte e
  página) e injetados no prompt.
- **System prompt** instrui o modelo a responder **só** com base no contexto,
  citar as fontes e admitir explicitamente quando a informação não existe nos
  documentos (anti-alucinação).
- Modelo: **Claude Haiku** (rápido e econômico para Q&A sobre contexto).

---

## 📊 Avaliação dos resultados

Avaliação automática com **8 perguntas reais de alunos** (`python app.py evaluate`):

| Métrica                 | Valor |
| ----------------------- | ----- |
| Perguntas testadas      | 8     |
| Erros de execução       | 0     |
| **Keyword recall médio**| **0.80** |
| **Taxa de fonte correta** | **0.75** |

- **Keyword recall**: proporção de termos-chave esperados que aparecem na
  resposta.
- **Taxa de fonte correta**: proporção de perguntas em que o documento
  esperado apareceu entre os chunks recuperados.

---

## 🐞 Análise de erros

Casos em que o sistema falhou ou ficou abaixo do esperado, com a causa raiz:

| Pergunta | Sintoma | Causa provável |
| -------- | ------- | -------------- |
| Carga horária total de Ciência da Computação | Modelo respondeu que não encontrou | Os chunks recuperados eram sobre **atividades complementares** (que também citam carga horária), não a tabela de integralização. Problema de **granularidade do chunk** — a carga horária total está numa tabela densa que não casou bem com a pergunta. |
| Prazos de matrícula 2026.2 | `source_match=False` | O calendário correto foi recuperado e a resposta estava certa, mas o documento **"Edital de Matrícula"** (esperado no ground truth) não entrou no top-5. O conteúdo de prazos está mais explícito no calendário do que no edital. |
| Aproveitamento de disciplinas de outra instituição | `source_match=False` | Informação **genuinamente ausente** nos documentos indexados. O sistema acertou ao admitir a lacuna (comportamento desejado), mas o ground truth apontava outro documento. |
| Normas para o TCC | `keyword_recall=0.33`, score baixo (0.59) | **Descasamento de vocabulário**: a pergunta usa "TCC/monografia", enquanto os PPCs falam em "Trabalho de Conclusão de Curso" de forma esparsa, sem um capítulo dedicado. A recuperação trouxe trechos fracos. |

### Possíveis melhorias
- **Chunking semântico** (por seção/título) em vez de janela fixa, para manter
  tabelas e regras inteiras no mesmo chunk.
- **Re-ranking** dos resultados (cross-encoder) após a busca vetorial.
- **Expansão de consulta** com sinônimos (ex.: TCC ↔ monografia ↔ trabalho de
  conclusão) para reduzir o descasamento de vocabulário.
- **Embedding multilíngue maior** (ex.: e5-large) para ganho de recall em PT.
- Aumentar `n_results` e filtrar por documento quando a pergunta cita o curso.

---

## 🧰 Tecnologias

- **Python 3.11**
- **pdfplumber** — extração de PDF
- **ChromaDB** — banco vetorial
- **all-MiniLM-L6-v2** — modelo de embeddings
- **Gemini** — geração das respostas
- **Typer + Rich** — interface de linha de comando

---

## 📄 Documentos indexados

PPCs (Computação, Pedagogia, História, Geografia, Direito), manuais de estágio
supervisionado, manual do aluno, edital de matrícula e calendário acadêmico
2026.2 — todos da UERR.
