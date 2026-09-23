# Local Embedding Models Performance Benchmark

Benchmark modular de alta performance para modelos de embedding locais com **FastAPI** e interface CLI, desenvolvido para avaliar latência, vazão (throughput), pegada de memória (RAM e VRAM GPU) e integridade dos vetores gerados a partir de documentos reais (**PDF, DOCX, DOC, ODT, TXT**).

Inspirado na arquitetura modular do projeto [`document_anonymizer`](file:///home/kennymar_inside/Documentos/document_anonymizer).

---

## Modelos Suportados por Padrão

| Modelo | Identificador Hugging Face | Dimensão | Max Seq Length | Particularidades Técnicas |
| :--- | :--- | :--- | :--- | :--- |
| **BGE-M3** | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | 1024 | 8192 | Multilíngue, suporte a contextos extensos. |
| **Qwen3-Embedding-8B** | [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 4096 | 32768 | Modelo denso de 8B parâmetros, alta capacidade semântica, suporte a bfloat16. |
| **Multilingual-E5-Large** | [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 512 | Requer prefixo (`passage: `) para indexação de documentos. |
| **Legal-BERTimbau-base** | [`rufimelo/Legal-BERTimbau-base`](https://huggingface.co/rufimelo/Legal-BERTimbau-base) | 768 | 512 | BERT treinado no domínio jurídico brasileiro com mean-pooling. |

---

## Métricas Avaliadas

Para cada modelo e para cada texto/chunk do documento:
1. **Tempo e Latência**:
   - **Tempo Total de Inferência** (\(s\) e \(ms\)).
   - **Latência Média por Chunk** (\(ms\)).
   - **Latência Mínima e Máxima por Chunk** (\(ms\)).
   - **Cold Start (Load Time)**: Tempo de carregamento do modelo na memória (\(s\)).
   - **Tempo de Warm-up**: Duração da primeira inferência de aquecimento (\(ms\)).
2. **Throughput (Vazão)**:
   - **Tokens por segundo** (\(tokens/s\)).
   - **Caracteres por segundo** (\(chars/s\)).
   - **Palavras por segundo** (\(words/s\)).
   - **Chunks por segundo** (\(chunks/s\)).
3. **Hardware & Memória**:
   - **RAM do Processo**: Consumo inicial, final e \(\Delta\) RAM (\(MB\)).
   - **VRAM da GPU (CUDA)**: Memória alocada, reservada e pico de VRAM (\(MB\)).
4. **Integridade Numérica**:
   - Dimensão efetiva do vetor.
   - Norma \(L_2\) média (verificação de normalização).
   - Detecção de anomalias numéricas (`NaN` ou `Inf`).

---

## Estrutura do Projeto

```
embedding_models_performance_test/
├── pyproject.toml                     # Dependências Poetry
├── README.md                          # Documentação detalhada
├── conftest.py                        # Configuração Pytest
├── app/
│   ├── main.py                        # Ponto de entrada FastAPI com rotas e healthcheck
│   ├── api/
│   │   └── endpoints/
│   │       └── v1/
│   │           ├── benchmark.py       # POST /api/v1/benchmark/run e /vectorize
│   │           └── models.py          # GET /api/v1/models (listagem e detalhes)
│   ├── core/
│   │   ├── config.py                  # Configurações globais e modelos disponíveis
│   │   ├── extractors/
│   │   │   └── file_extractor.py      # Extração unificada (PDF, DOCX, DOC, ODT, TXT)
│   │   ├── chunkers/
│   │   │   └── text_chunker.py        # Chunking (paragraph, fixed, sentence)
│   │   └── models/
│   │       ├── base.py                # BaseEmbeddingModel (ABC)
│   │       ├── registry.py            # ModelRegistry (Fábrica e registro modular)
│   │       ├── bge_m3.py              # Classe BGE-M3
│   │       ├── qwen3.py               # Classe Qwen3-8B
│   │       ├── e5_large.py            # Classe Multilingual-E5-Large
│   │       ├── legal_bertimbau.py     # Classe Legal-BERTimbau-base
│   │       └── generic_st.py          # Wrapper genérico SentenceTransformers
│   ├── schemas/
│   │   └── benchmark.py               # Schemas Pydantic (validação e serialização)
│   └── services/
│       └── benchmark_service.py       # Orquestrador de benchmark e isolamento de VRAM
├── benchmark/
│   ├── config.py                      # Configurações do benchmark CLI e gráficos
│   ├── evaluator.py                   # Avaliador de arquivos no disco
│   ├── visualizer.py                  # Gráficos em PNG (latência, vazão, memória, dashboard)
│   ├── run_benchmark.py               # Script CLI executável
│   └── data/
│       ├── sample_documents/          # Documentos de teste
│       └── results/                   # Relatórios JSON, TXT e gráficos gerados
└── tests/
    ├── test_extractors.py             # Testes de extração para todos os formatos
    ├── test_chunker.py                # Testes de estratégias de chunking
    ├── test_models.py                 # Testes unitários do ModelRegistry e classes base
    └── test_api.py                    # Testes de integração da API FastAPI
```

---

## Instalação e Configuração

Certifique-se de ter o Python 3.11+ e o Poetry instalados.

```bash
# Clone ou acesse o diretório do projeto
cd /home/kennymar_inside/Documentos/embedding_models_performance_test

# Instale as dependências
poetry install
```

---

## Modularidade: Como Adicionar ou Remover Modelos

A adição de qualquer novo modelo de embedding local é feita através de **duas formas simples**:

### Forma 1: Apenas adicionando no `app/core/config.py` (Sem escrever código Python)
Caso o modelo seja compatível com `SentenceTransformers`, basta adicionar uma nova entrada ao dicionário `AVAILABLE_MODELS`:

```python
AVAILABLE_MODELS["meu-novo-modelo"] = {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "display_name": "MiniLM L6 v2",
    "class_key": "generic_st",
    "max_seq_length": 256,
    "default_batch_size": 16,
    "device": DEFAULT_DEVICE,
    "torch_dtype": "float32",
    "normalize_embeddings": True,
    "enabled": True,
    "description": "Modelo leve e rápido para testes.",
}
```

### Forma 2: Criando uma classe especializada (Para modelos com particularidades)
1. Crie um arquivo em `app/core/models/meu_modelo.py` herdando de `BaseEmbeddingModel`:
```python
from app.core.models.base import BaseEmbeddingModel
from app.core.models.registry import ModelRegistry

@ModelRegistry.register("meu_modelo_custom")
class MeuModeloCustom(BaseEmbeddingModel):
    def _load_model(self):
        # Lógica de carregamento específica
        ...
    def _encode_batch(self, texts, batch_size):
        # Lógica de codificação customizada
        ...
```
2. Defina `class_key: "meu_modelo_custom"` na configuração em `config.py`.

---

## Executando a API FastAPI
## Interface Web Interativa & API FastAPI

Inicie o servidor HTTP com Uvicorn:

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

A documentação interativa Swagger estará acessível em:
### Painel Web com Caixas de Seleção (Dashboard)
Acesse no seu navegador:
**`http://localhost:8000/`**

O painel oferece:
- **Área Drag & Drop** para upload de documentos (`.pdf`, `.docx`, `.doc`, `.odt`, `.txt`).
- **Lista interativa com Caixas de Seleção (Checkboxes)** para cada modelo com botões de *Selecionar Todos* e *Desmarcar Todos*.
- **Parâmetros ajustáveis** (Batch size, Chunk size, Estratégia de chunking).
- **Visualização imediata**: Tabela detalhada de métricas, ranking com medalhas (mais rápido, maior throughput, menor VRAM) e gráficos comparativos gerados em tempo real.
- **Download com 1 clique** dos relatórios estruturados em JSON.

### Documentação Swagger Interativa
Acesse em:
**`http://localhost:8000/docs`**

### Endpoints Disponíveis:

#### 1. `POST /api/v1/benchmark/run`
Envia um documento e executa o benchmark em todos os modelos selecionados.
- **Form Data**:
  - `file`: Arquivo a ser enviado (`.pdf`, `.docx`, `.doc`, `.odt`, `.txt`).
  - `models` *(opcional)*: Lista separada por vírgula (ex: `bge-m3,multilingual-e5-large,legal-bertimbau-base`).
  - `models` *(opcional)*: Caixas de seleção com múltiplos valores `models` ou string separada por vírgula (ex: `bge-m3,multilingual-e5-large,legal-bertimbau-base`).
  - `batch_size` *(opcional)*: Tamanho do lote.
  - `chunk_size` *(opcional, padrão 500)*: Tamanho do chunk em caracteres.
  - `chunk_strategy` *(opcional, padrão "paragraph")*: `"paragraph"`, `"fixed"`, `"sentence"`.
  - `save_report` *(opcional, padrão True)*: Salva relatório em disco.

#### 2. `POST /api/v1/benchmark/vectorize`
Vetoriza o documento com um único modelo e retorna métricas + vetores.
- **Form Data**:
  - `file`: Documento a ser vetorizado.
  - `model_name`: Identificador do modelo (ex: `bge-m3`).
  - `include_vectors`: Se `True`, retorna os arrays numéricos no JSON.

#### 3. `GET /api/v1/models`
Lista todos os modelos de embedding locais cadastrados no sistema, com status e configurações.

#### 4. `GET /health`
Informa status da API, suporte a CUDA, nome da GPU e VRAM total.

---

## Executando o Benchmark via CLI

Para rodar via linha de comando e gerar gráficos em alta resolução:

```bash
# Executa no arquivo de exemplo padrão com todos os modelos
poetry run python benchmark/run_benchmark.py

# Ou especifique um documento e modelos específicos:
poetry run python benchmark/run_benchmark.py \
    --document benchmark/data/sample_documents/exemplo_juridico.txt \
    --models bge-m3,multilingual-e5-large,legal-bertimbau-base \
    --batch-size 8 \
    --chunk-size 500 \
    --strategy paragraph
```

Os resultados são salvos em `benchmark/data/results/<TIMESTAMP>/`:
- `summary.json`: Métricas estruturadas completas.
- `summary.txt`: Relatório descritivo formatado.
- `latency_comparison.png`: Gráfico comparativo de tempo e latência.
- `throughput_comparison.png`: Gráfico de tokens/s e chunks/s.
- `memory_comparison.png`: Gráfico de consumo de RAM e pico de VRAM.
- `metrics_dashboard.png`: Painel 2x2 com todas as métricas consolidadas.

---

## Executando os Testes Automatizados

O projeto conta com suite de testes completa via Pytest:

```bash
poetry run pytest -v
```

