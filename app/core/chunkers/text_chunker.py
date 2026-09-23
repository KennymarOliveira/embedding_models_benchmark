import re
from typing import List, Dict, Any, Literal


def estimate_tokens(text: str) -> int:
    """Estimativa de tokens para texto em português/multilíngue."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(round((words * 1.3 + len(text) / 4) / 2)))


def split_into_paragraphs(text: str) -> List[str]:
    """Divide o texto por quebras de linha duplas ou parágrafos."""
    paras = re.split(r"\n\s*\n", text)
    cleaned = [p.strip() for p in paras if p.strip()]
    return cleaned if cleaned else [text.strip()] if text.strip() else []


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: Literal["paragraph", "fixed", "sentence"] = "paragraph",
) -> List[Dict[str, Any]]:
    """
    Divide um texto longo em chunks com base na estratégia especificada.

    Retorna uma lista de dicionários contendo:
    - chunk_id: índice do chunk (0-based)
    - text: conteúdo textual do chunk
    - char_count: número de caracteres
    - word_count: número de palavras
    - est_token_count: estimativa de número de tokens
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    raw_chunks: List[str] = []

    if strategy == "paragraph":
        paragraphs = split_into_paragraphs(text)
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > chunk_size and current_chunk:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = p_len
            else:
                current_chunk.append(p)
                current_len += p_len + 2

        if current_chunk:
            raw_chunks.append("\n\n".join(current_chunk))

    elif strategy == "fixed":
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(text), step):
            segment = text[i : i + chunk_size].strip()
            if segment:
                raw_chunks.append(segment)
            if i + chunk_size >= len(text):
                break

    elif strategy == "sentence":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = []
        current_len = 0

        for s in sentences:
            s_len = len(s)
            if current_len + s_len > chunk_size and current_chunk:
                raw_chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_len = s_len
            else:
                current_chunk.append(s)
                current_len += s_len + 1

        if current_chunk:
            raw_chunks.append(" ".join(current_chunk))

    else:
        raw_chunks = [text]

    structured_chunks = []
    for idx, c in enumerate(raw_chunks):
        if not c.strip():
            continue
        structured_chunks.append(
            {
                "chunk_id": idx,
                "text": c,
                "char_count": len(c),
                "word_count": len(c.split()),
                "est_token_count": estimate_tokens(c),
            }
        )

    return structured_chunks

