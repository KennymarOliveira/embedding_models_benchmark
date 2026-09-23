from app.core.chunkers.text_chunker import chunk_text, estimate_tokens, split_into_paragraphs


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("olá mundo") >= 2
    tokens = estimate_tokens("Este é um texto um pouco mais longo para testar a estimativa.")
    assert tokens > 5


def test_split_into_paragraphs():
    text = "Parágrafo um.\n\nParágrafo dois.\n\n\nParágrafo três."
    paras = split_into_paragraphs(text)
    assert len(paras) == 3
    assert paras[0] == "Parágrafo um."
    assert paras[1] == "Parágrafo dois."
    assert paras[2] == "Parágrafo três."


def test_chunk_text_paragraph():
    text = "Primeiro parágrafo longo.\n\nSegundo parágrafo curto.\n\nTerceiro parágrafo adicional."
    chunks = chunk_text(text, chunk_size=30, strategy="paragraph")
    assert len(chunks) >= 2
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "char_count" in c
        assert "word_count" in c
        assert "est_token_count" in c


def test_chunk_text_fixed():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2, strategy="fixed")
    assert len(chunks) == 3
    assert chunks[0]["text"] == "ABCDEFGHIJ"
    assert chunks[1]["text"] == "IJKLMNOPQR"
    assert chunks[2]["text"] == "QRSTUVWXYZ"


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
