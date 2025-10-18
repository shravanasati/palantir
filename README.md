# Palantir

<p align="center"> 
	<img src="palantir.png" height="300px">
</p>

A powerful keyword search engine for movie data using advanced information retrieval techniques.

## Overview

Palantir is a Python-based search system that indexes and searches a movie dataset using multiple retrieval methods:
- **Keyword Search**: Simple keyword matching in movie titles
- **TF-IDF Search**: Relevance ranking based on term frequency-inverse document frequency
- **BM25 Search**: Advanced ranking using the Okapi BM25 algorithm

The system uses an inverted index for efficient querying and supports caching for improved performance.

## Features

- **Inverted Index**: Fast lookup of documents containing specific terms
- **Text Preprocessing**: Case normalization, tokenization, stopword removal, and Porter stemming
- **Multiple Ranking Algorithms**: TF-IDF and BM25 for relevance scoring
- **Caching**: Persistent storage of index data for quick reloads
- **CLI Interface**: Command-line tool for querying and analyzing search metrics

## Installation

1. `uv venv`

2. `source ./.venv/bin/activate`

3. `uv sync`


## Usage

### Build the Index

First, build and cache the inverted index from the dataset:

```bash
python cli/keyword_search_cli.py build
```

### Search with BM25

```bash
python cli/keyword_search_cli.py bm25search "your query" [limit]
```

Example:
```bash
python cli/keyword_search_cli.py bm25search "terminator" 5
```

### Other Commands

**Basic keyword search:**
```bash
python cli/keyword_search_cli.py search "query"
```


## Algorithm Parameters

### BM25 Configuration

- **K1** (default: 1.5): Controls term frequency saturation. Higher values = more impact from increased term frequency
- **B** (default: 0.75): Controls length normalization. 0 = no normalization, 1 = full normalization

## Dataset

The project uses a movie dataset [(`data/movies.json`)](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json) with fields:
- `id`: Unique movie identifier
- `title`: Movie title
- `description`: Movie description

