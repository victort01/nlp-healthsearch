# HealthSearch

Busca híbrida para um corpus didático de saúde, combinando BM25, embeddings multilíngues reais, RRF ponderado e Cross-Encoder opcional nos três primeiros candidatos. Os textos são material acadêmico gerado com IA, não orientações clínicas.

Projeto da disciplina **Tendências em Ciência da Computação — UNIPÊ**, professor Me. Ricardo Roberto de Lima. Os enunciados locais e a orientação complementar sobre criação de dados com IA definem o escopo.

## Executar

Python 3.12. Crie um ambiente virtual e ative-o:

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
# Linux ou macOS
source .venv/bin/activate
```

Depois de ativar o ambiente:

```bash
python -m pip install -r requirements.txt
python -m streamlit run healthsearch_app.py
```

As alternativas de ativação acima dependem do sistema operacional; execute apenas a correspondente. O aplicativo abre no endereço local indicado pelo Streamlit.

## Testes

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Os testes verificam cálculos e casos de borda; os testes de interface usam `streamlit.testing.v1.AppTest`. O CI não baixa modelos. As avaliações reais e os artefatos produzidos ficam em `resultados/`, separados dos testes rápidos.

## Modelos e primeiro uso

O primeiro uso baixa pesos públicos do Hugging Face, que podem ocupar centenas de MB por modelo. Inferência local em CPU, sem chave de API. Instale PyTorch para CPU com `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` antes das demais dependências se quiser evitar a distribuição com suporte a GPU. Os pesos não são versionados. Falhas de download são exibidas; não há simulação silenciosa de embeddings.

## Entrega e arquitetura

- `healthsearch_app.py`: app completo, seis trechos originais obrigatórios e textos expandidos incorporados.
- `relatorio_tecnico.pdf`: relatório de duas páginas, com gráfico de ranks.
- `corpus/`: seis documentos de duas páginas. PDFs são apoio para leitura; não são dependências do app.
- `avaliar.py`: doze consultas didáticas, gráfico e teste real do bônus.

Execute `python avaliar.py` para regenerar resultados (inclui o download do Cross-Encoder).

1. BM25 usa os documentos completos normalizados, com remoção de stopwords. Hífens internos preservam códigos completos como `CÓD-ECG-12D`.
2. O encoder `paraphrase-multilingual-MiniLM-L12-v2` indexa janelas que respeitam seu limite de tokens, com overlap de 24 tokens. O melhor cosseno entre trechos define o score do documento e fornece evidência textual.
3. `RRF(D) = α/(60 + rank_BM25) + (1−α)/(60 + rank_semântico)`, com posições iniciando em 1 e desempate por ID. α=1 preserva a ordem léxica; α=0, a semântica.
4. O bônus `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reordena somente os Top-3 do RRF usando o trecho recuperado. Logits não são probabilidades nem scores RRF.

Sliders: k₁ de 0 a 3, padrão 1,2; b de 0 a 1, padrão 0,75; α de 0 a 1, padrão 0,5. O caso k₁=0 implementa o limite binário para evitar 0/0 da biblioteca.

### Limitações

A fusão segue exatamente a fórmula pedida, incluindo documentos com BM25 zero: nesses empates, a ordem por ID pode influenciar o resultado. O app informa isso. RRF não é uma probabilidade e não garante superioridade para toda consulta. Tempos incluem aquecimento/download no primeiro uso. As doze consultas são exemplos exploratórios com referência didática, não benchmark cego ou validação clínica; o código ECG é compartilhado por dois documentos.

## Origem e referências

A implementação foi preparada com auxílio de IA. Os textos expandidos e a base sintética também foram produzidos com IA para fins didáticos. As propostas anteriores do próprio aluno foram usadas como contexto; nenhuma alegação de execução antiga foi reutilizada como evidência de teste desta versão.

Documentação: [Streamlit](https://docs.streamlit.io/), [Sentence Transformers](https://sbert.net/), [rank_bm25](https://github.com/dorianbrown/rank_bm25), [RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

## Verificação desta entrega

Em 27/09/2026, **12 testes rápidos passaram** em Python 3.12 e Windows. Evidências: [testes](resultados/testes.xml) e [ambiente](resultados/ambiente.json). O workflow do GitHub repete os testes rápidos em Linux.

A [integração com modelos reais](resultados/integracao.json) também passou. Reproduza com `python verificar_integracao.py` após instalar as dependências de desenvolvimento.

BM25, semântico e RRF tiveram 10/12 acertos Top-1 nas [consultas exploratórias](resultados/comparacao_ranks.csv). O empate agregado não elimina diferenças por consulta. As [referências do corpus](corpus/REFERENCIAS.md) complementam os marcadores dos PDFs.

Para regenerar os PDFs a partir das fontes locais, instale `requirements-dev.txt` e execute `python gerar_pdfs.py`. O conteúdo do relatório está em `relatorio.json`; ao refazer avaliações com outros parâmetros, revise esse texto antes de gerar o PDF.
