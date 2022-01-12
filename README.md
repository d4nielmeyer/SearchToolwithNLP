# SearchToolwithNLP *[work in progress]*
This project is about building an efficient and smart search engine for fast access to the CDC’s document database. The goal is to improve the CDC’s ability to handle future pandemics, with the capability to aggregate and search unstructured text data from records of earlier outbreaks.

## Project Outline:

**1. Text Search with Spacy and scikit-learn.**
  - Preprocess the data with spacy.
  - Compute TF-IDF tables and apply term frequency search to them.
  - Calculate cosine similarity with scikit-learn.
  - Build an inverted index, an essential element of a search engine.

**2. Implement Semantic Search with ML and BERT.**
  - Build a search engine using FAISS similarity search library and a pre-trained DistilBERT model from Transformers.
  - Implement a search engine using sentence-transformers and FAISS (using SBERT instead of a base BERT model).
  - Create a question answering agent powered by sentence-transformers, FAISS similarity search engine, and a BERT model for question answering.
  
**3. Build a Search API with Elasticsearch**
  - Setting up and indexing an Elasticsearch Docker Container.
  - Querying an Elasticsearch index.
  - Boosting search relevance of Elasticsearch results with BERT.
  - Creating a Flask API serving a BERT model for semantic search.
  
**4. UI for a Search API with Flask and Bootstrap**
  - ...

## Tech Stack:

- Numpy: Basic operations
- Scikit-learn: Implement a TD-IDF search and an inverted index search
- Spacy: Perform essential natural language processing steps
- ...
