# Confluence Q&A with LLMs using ChromaDB

This project provides a toolkit to extract data from your Confluence wiki, process it, store it in a ChromaDB vector database, and use it to answer questions with Large Language Models (LLMs). It enables you to build a Retrieval-Augmented Generation (RAG) system based on your Confluence knowledge base.

## Features

- **Confluence Data Extraction:** Fetches pages from specified Confluence spaces using the API.
- **Content Processing:** Cleans HTML, converts it to Markdown, and preserves page metadata.
- **Vector Database Integration:** Chunks documents, generates embeddings (using sentence-transformers), and stores them in a persistent ChromaDB database (`chroma_db/`).
- **LLM Interaction Example:** Includes an example script (`vector_search_example.py`) demonstrating how to query the ChromaDB index and use the retrieved context with an LLM to answer questions.
- **Flexible Output:** Saves extracted data as individual Markdown files (`confluence_data/pages/`) and a consolidated JSON (`confluence_data/confluence_pages.json`).

## Requirements

- Python 3.7+
- Access to a Confluence instance (Username/Email and API Token)
- Dependencies listed in `requirements.txt`:
  - `requests`
  - `beautifulsoup4`
  - `html2text`
  - `python-dotenv`
  - `langchain`
  - `chromadb`
  - `pyyaml`
  - `sentence-transformers`
  - `numpy`

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file:** Copy the `.env.example` file to `.env`:

    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** with your Confluence details:

    ```dotenv
    CONFLUENCE_BASE_URL=https://your-instance.atlassian.net/wiki
    CONFLUENCE_USERNAME=your-email@example.com
    CONFLUENCE_API_TOKEN=your-api-token

    # Optional: Configure output directory for raw data
    OUTPUT_DIR=confluence_data

    # Optional: Configure ChromaDB path (defaults to ./chroma_db)
    # CHROMA_DB_PATH=./chroma_db

    # Optional: Configure embedding model
    # EMBEDDING_MODEL=all-MiniLM-L6-v2

    # Optional: OpenAI API Key if using OpenAI models in examples
    # OPENAI_API_KEY=your-openai-api-key
    ```

    - **Get a Confluence API Token:**
      1.  Log in to `https://id.atlassian.com/manage-profile/security/api-tokens`
      2.  Click "Create API token"
      3.  Give it a label (e.g., "LLM Extractor") and copy the token.

## Usage

1.  **Extract Confluence Data:**
    Run the extractor script. This will fetch pages, process them, and save them to the `OUTPUT_DIR` (default: `confluence_data`).

    ```bash
    python confluence_extractor.py [options]
    ```

    - **Options:**
      - `--spaces KEY1 KEY2`: Specify space keys to extract (default: all accessible spaces).
      - `--output-dir`: Override the output directory set in `.env`.
      - _See `python confluence_extractor.py --help` for all options._

2.  **Build Vector Index & Query with LLM:**
    Run the vector search example script. This script will:

    - Load the processed Markdown files from `confluence_data/pages/`.
    - Chunk the documents.
    - Generate embeddings using the specified sentence-transformer model.
    - Create or load a persistent ChromaDB index in `chroma_db/`.
    - Take your query, search the index for relevant document chunks.
    - (If an LLM is configured) Send the query and retrieved context to an LLM to generate an answer.

    ```bash
    python vector_search_example.py --query "Your question about Confluence content?"
    ```

    - **Options:**
      - `--query`: The question you want to ask.
      - `--embedding-model`: Override the embedding model (default: `all-MiniLM-L6-v2`).
      - `--chroma-db-path`: Specify the path to the Chroma database directory.
      - `--top-k`: Number of relevant chunks to retrieve (default: 5).
      - _See `python vector_search_example.py --help` for potentially more options related to LLM choice._

## How it Works (RAG Flow)

1.  **Extraction:** `confluence_extractor.py` fetches and preprocesses data.
2.  **Loading:** `vector_search_example.py` loads the Markdown files.
3.  **Chunking:** Documents are split into smaller, manageable chunks.
4.  **Embedding:** Each chunk is converted into a numerical vector representation (embedding) using a sentence-transformer model.
5.  **Indexing:** Embeddings and corresponding text chunks are stored in a ChromaDB vector database for efficient similarity searching.
6.  **Retrieval:** When you provide a query, it's embedded, and ChromaDB finds the most similar chunks from the index.
7.  **Augmentation & Generation:** The original query and the retrieved chunks (context) are passed to an LLM, which generates an answer based on the provided information.

## Troubleshooting

- **Rate Limiting:** Confluence API might have rate limits. The extractor doesn't currently handle this explicitly; consider adding delays if needed.
- **Authentication:** Double-check your `.env` file for correct Confluence URL, username, and a valid API token. Ensure the token has read permissions.
- **ChromaDB Errors:** Ensure `chroma_db/` is writable. Check ChromaDB documentation for specific issues.
- **Dependencies:** Make sure all dependencies in `requirements.txt` are installed correctly within your virtual environment.

## License

MIT
