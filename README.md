# LLM-Librarian

This is a repo for using LLMs as a librarian to classify documents and extract type-specific metadata from them before filing them into an OpenSearch database and a vector store run by RAGatouille/Byaldi using a Flask server I wrote. Currently it only works on PDFs, which are placed in the pdfs folder. 

This will be used as the base of a larger RAG system that search and research agents use as their knowledge store.

To run this:
Set your GEMINI_API_KEY environment variable (or other api key that is supported by PydanticAI)

(I usually run tmux here)

cd DwarfInTheFlask

uv run flask_server.py

Navigate back to root (or exit tmux)

uv run process_all.py

NOTES:
-This uses OCR on the first page of the document to increase reliability of title extraction for books, which sometimes aren't actually included in the text of the PDF.
-Metadata extraction will fall back on using a partial page set if something goes wrong with using the full pdf, such as context lengths for the LLM. By default it uses gemini to avoid this issue, but I want it to not rely on Gemini.
-Chunk text extraction for OpenSearch indexing uses PyMuPDF. This is separate from the OCR text because when using partial pages for metadata extraction it needs to make sure to use all the text for chunk indexing.