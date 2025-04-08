#!/usr/bin/env python3
"""
Advanced example demonstrating how to use extracted Confluence data with a vector database
for more effective retrieval before sending to an LLM.

This example uses FAISS for vector storage and sentence-transformers for embeddings.
"""

import os
import json
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import required packages, but don't fail if they're not installed
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    logger.warning("FAISS or sentence-transformers not installed. Vector search will not be available.")
    VECTOR_SEARCH_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not installed. Using mock LLM responses.")
    OPENAI_AVAILABLE = False


def load_confluence_data(file_path: str) -> List[Dict[str, Any]]:
    """Load Confluence data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of page dictionaries
    """
    logger.info(f"Loading Confluence data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} pages")
    return data


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a good break point
        if end < len(text):
            # Try to find a period, question mark, or exclamation point followed by a space
            for i in range(end - 1, start + chunk_size // 2, -1):
                if i < len(text) and text[i] in ['.', '!', '?'] and (i + 1 == len(text) or text[i + 1] == ' '):
                    end = i + 1
                    break
        else:
            end = len(text)
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move to the next chunk with overlap
        start = end - overlap
    
    return chunks


def prepare_data_for_indexing(pages: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Prepare data for indexing by chunking the content.
    
    Args:
        pages: List of page dictionaries
        
    Returns:
        Tuple of (list of text chunks, list of chunk metadata)
    """
    all_chunks = []
    all_metadata = []
    
    for page in pages:
        # Split the content into chunks
        chunks = chunk_text(page['content'])
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            
            # Store metadata for this chunk
            metadata = {
                'page_id': page['id'],
                'page_title': page['title'],
                'space_key': page['space_key'],
                'url': page['url'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            all_metadata.append(metadata)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks, all_metadata


def create_vector_index(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Any, Any]:
    """Create a FAISS vector index from text chunks.
    
    Args:
        chunks: List of text chunks
        model_name: Name of the sentence-transformer model to use
        
    Returns:
        Tuple of (FAISS index, sentence transformer model)
    """
    if not VECTOR_SEARCH_AVAILABLE:
        raise ImportError("FAISS and sentence-transformers are required for vector search")
    
    logger.info(f"Creating vector index with model: {model_name}")
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Create embeddings
    logger.info("Creating embeddings for chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Create the index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    logger.info(f"Created index with {index.ntotal} vectors of dimension {dimension}")
    return index, model


def search_vector_index(
    query: str, 
    index: Any, 
    model: Any, 
    metadata: List[Dict[str, Any]], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Search the vector index for relevant chunks.
    
    Args:
        query: Search query
        index: FAISS index
        model: Sentence transformer model
        metadata: List of chunk metadata
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with chunk text and metadata
    """
    # Create query embedding
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search the index
    scores, indices = index.search(query_embedding, top_k)
    
    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 means no result
            results.append({
                'score': float(scores[0][i]),
                'metadata': metadata[idx],
                'text': chunks[idx]
            })
    
    logger.info(f"Found {len(results)} relevant chunks")
    return results


def create_prompt(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Create a prompt for the LLM using the query and search results.
    
    Args:
        query: User query
        search_results: List of search result dictionaries
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Question: {query}\n\n"
    prompt += "Context from Confluence:\n\n"
    
    for i, result in enumerate(search_results, 1):
        metadata = result['metadata']
        prompt += f"--- Document {i}: {metadata['page_title']} (Chunk {metadata['chunk_index'] + 1}/{metadata['total_chunks']}) ---\n"
        prompt += result['text'] + "\n\n"
    
    prompt += "Based on the above context only, please answer the question concisely and accurately."
    return prompt


def query_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Query OpenAI's API with the given prompt.
    
    Args:
        prompt: The prompt to send to the API
        model: The model to use
        
    Returns:
        The response from the API
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not available. Returning mock response.")
        return "This is a mock response. Please install the OpenAI package to use the actual API."
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    logger.info(f"Querying OpenAI API with model: {model}")
    
    # Log the variables we're using for debugging
    logger.debug(f"Using model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on Confluence documentation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=500    # Limit response length
    )
    
    return response.choices[0].message.content


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description='Vector search example for Confluence data')
    parser.add_argument('--data', default='confluence_data/confluence_pages.json', 
                        help='Path to the Confluence data JSON file')
    parser.add_argument('--query', required=True, help='The query to answer')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', 
                        help='Sentence transformer model to use for embeddings')
    parser.add_argument('--top-k', type=int, default=5, 
                        help='Number of top results to retrieve')
    
    args = parser.parse_args()
    
    try:
        # Check if vector search is available
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Vector search is not available. Please install FAISS and sentence-transformers.")
            print("To install required packages:")
            print("pip install faiss-cpu sentence-transformers")
            return 1
        
        # Load the Confluence data
        pages = load_confluence_data(args.data)
        
        # Prepare data for indexing
        global chunks  # Make chunks available to search function
        chunks, metadata = prepare_data_for_indexing(pages)
        
        # Create vector index
        index, model = create_vector_index(chunks, args.embedding_model)
        
        # Search for relevant chunks
        search_results = search_vector_index(
            args.query, 
            index, 
            model, 
            metadata, 
            args.top_k
        )
        
        if not search_results:
            print("No relevant information found in the Confluence data.")
            return 0
        
        # Create a prompt for the LLM
        prompt = create_prompt(args.query, search_results)
        
        # Print the sources being used
        print("\nSources used:")
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            print(f"{i}. {metadata['page_title']} (Score: {result['score']:.4f})")
        
        # Query the LLM
        response = query_openai(prompt, args.model)
        
        # Print the response
        print("\nResponse:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 