# Rastreabilidade do enunciado

Os documentos da atividade definem os requisitos desta implementação. A orientação complementar do professor, comunicada pelo aluno, determina criação/ampliação dos textos com IA.

| Requisito | Evidência |
| --- | --- |
| Corpus de seis documentos | CORPUS preserva frases; PAGINAS_EXPANDIDAS amplia cada tema |
| Normalização e stopwords | tokenizar() |
| BM25 com k₁ e b nos intervalos pedidos | Sidebar e scores_bm25(), inclusive k₁=0 |
| Embeddings e cosseno | encoder(), dividir_corpus(), pesquisar() |
| Fórmula ponderada RRF k=60 | rrf(), ranks() e parcelas exibidas |
| Quatro abas exigidas | Léxico, Semântico, Híbrido RRF, Matriz Comparativa |
| Gráfico e métricas | Aba Matriz Comparativa e resultados/ |
| Cross-Encoder Top-3 bônus | reranquear() e checkbox |
| Arquivo único e PDF ≤2 páginas | healthsearch_app.py e relatorio_tecnico.pdf |
| Equipe | Divisão proposta no relatório, não alegação de histórico |
