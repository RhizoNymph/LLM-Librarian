# LLM-Librarian

This is a repo for using LLMs as a librarian to classify documents and extract type-specific metadata from them before filing them into an OpenSearch database. Currently it only works on PDFs, which are placed in the pdfs folder. 

This will be used as the base of a larger RAG system that search and research agents use as their knowledge store.

uv run process_all.py

TODO:
- add vector search indexing as well