# Palantir

<p align="center"> 
	<img src="palantir.png" height="300px">
</p>

A powerful hybrid search engine for movie data combining advanced information retrieval techniques, semantic search, and AI-powered query enhancement.

## Overview

Palantir is a Python-based search system that indexes and searches a movie dataset using multiple retrieval methods:
- **Keyword Search**: BM25 ranking with inverted index and text preprocessing
- **Semantic Search**: Vector embeddings with sentence transformers and chunking strategies
- **Hybrid Search**: Combines keyword and semantic search using weighted scoring or Reciprocal Rank Fusion (RRF)
- **Query Enhancement**: AI-powered query rewriting, spell correction, and expansion using Google Gemini
- **Reranking**: Cross-encoder and LLM-based reranking for improved relevance

## Features

### Search Capabilities
- **Inverted Index**: Fast lookup with TF-IDF and BM25 scoring
- **Text Preprocessing**: Case normalization, tokenization, stopword removal, and Porter stemming
- **Semantic Embeddings**: Using SentenceTransformer models (all-MiniLM-L6-v2)
- **Chunking**: Fixed-size and semantic chunking for better context retrieval
- **Hybrid Fusion**: Weighted combination and RRF for merging search results

### Advanced Features
- **Query Enhancement**: Spell correction, query rewriting, and expansion via Google Gemini
- **Reranking**: Cross-encoder (ms-marco-TinyBert-L2-v2) and LLM-based reranking
- **Caching**: Persistent storage of indexes and embeddings for quick reloads
- **Evaluation Framework**: Precision@k, Recall@k, and F1-score metrics with golden dataset

## Installation

1. Create and activate virtual environment:
```bash
uv venv
source ./.venv/bin/activate
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables (for query enhancement and reranking):
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

### Keyword Search

**Build the inverted index:**
```bash
python cli/keyword_search_cli.py build
```

**Search with BM25:**
```bash
python cli/keyword_search_cli.py bm25search "terminator" 5
```

**Other keyword commands:**
```bash
# Basic keyword search
python cli/keyword_search_cli.py search "query"

# Calculate IDF for a term
python cli/keyword_search_cli.py idf "action"

# Get term frequency in a document
python cli/keyword_search_cli.py tf 1 "movie"

# Get TF-IDF score
python cli/keyword_search_cli.py tfidf 1 "action"
```

### Semantic Search

**Basic semantic search:**
```bash
python cli/semantic_search_cli.py search "robot fighting humans" 5
```

**Chunked semantic search:**
```bash
python cli/semantic_search_cli.py search_chunked "space adventure" 5
```

**Generate embeddings:**
```bash
# Generate embeddings for all documents
python cli/semantic_search_cli.py verify_embeddings

# Generate chunked embeddings
python cli/semantic_search_cli.py embed_chunks

# Embed specific text
python cli/semantic_search_cli.py embed "example text"
```

**Text chunking:**
```bash
# Fixed-size chunking
python cli/semantic_search_cli.py chunk "your text here" --chunk-size 200 --overlap 40

# Semantic chunking
python cli/semantic_search_cli.py semantic_chunk "your text here" --max-chunk-size 4 --overlap 1
```

### Hybrid Search

**Weighted hybrid search:**
```bash
# Alpha controls keyword vs semantic weight (0-1)
python cli/hybrid_search_cli.py weighted_search "sci-fi movie" --limit 5 --alpha 0.6
```

**RRF (Reciprocal Rank Fusion) search:**
```bash
# Basic RRF search
python cli/hybrid_search_cli.py rrf_search "romantic comedy" --limit 5 --k 60

# With query enhancement
python cli/hybrid_search_cli.py rrf_search "scary bear movie" --limit 5 --enhance rewrite

# With reranking
python cli/hybrid_search_cli.py rrf_search "space adventure" --limit 5 --rerank

# With both enhancement and reranking
python cli/hybrid_search_cli.py rrf_search "that movie about bear" --limit 5 --enhance rewrite --rerank
```

**Query enhancement methods:**
- `spell`: Fix spelling errors
- `rewrite`: Rewrite query for better searchability
- `expand`: Add synonyms and related terms

### Evaluation

Run comprehensive evaluation with golden dataset:
```bash
python cli/evaluation_cli.py -k 5
```

This evaluates all search methods and combinations:
- Keyword Search (BM25)
- Semantic Search (Chunked)
- Hybrid Search (Weighted and RRF)
- Hybrid Search + Query Enhancement
- Hybrid Search + Reranking
- Hybrid Search + Enhancement + Reranking

## Algorithm Parameters

### BM25 Configuration
- **K1** (default: 1.5): Controls term frequency saturation. Higher values = more impact from term frequency
- **B** (default: 0.75): Controls length normalization. 0 = no normalization, 1 = full normalization

### Semantic Search
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Chunking**: Semantic chunking with 4 sentences per chunk and 1 sentence overlap
- **Similarity**: Cosine similarity between query and document embeddings

### Hybrid Search
- **Weighted**: Alpha parameter controls keyword vs semantic balance
- **RRF**: Reciprocal Rank Fusion with k=60 parameter

### Reranking
- **Cross-encoder**: ms-marco-TinyBert-L2-v2 for fast reranking
- **LLM**: Google Gemini for contextual reranking

## Dataset

The project uses a movie dataset [(`data/movies.json`)](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json) with fields:
- `id`: Unique movie identifier
- `title`: Movie title
- `description`: Movie description
