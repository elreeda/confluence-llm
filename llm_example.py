#!/usr/bin/env python3
"""
Example script demonstrating how to use extracted Confluence data with ChromaDB and LLMs.
This script loads Markdown files, chunks them, stores embeddings in ChromaDB,
and provides query functionality using free, open-source models.

Features:
- Conversational mode with history tracking
- Learning from user feedback
- Vector database for efficient semantic search
"""

import os
import argparse
import logging
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import required packages
try:
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependency: {str(e)}")
    logger.error("Please install required packages: pip install chromadb langchain sentence-transformers")
    DEPENDENCIES_AVAILABLE = False


def load_markdown_files(directory: str) -> List[Dict[str, Any]]:
    """Load Markdown files from a directory.
    
    Args:
        directory: Path to the directory containing Markdown files
        
    Returns:
        List of dictionaries with filename and content
    """
    logger.info(f"Loading Markdown files from {directory}")
    docs = []
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        
                        # Extract YAML front matter if present
                        metadata = {}
                        if content.startswith("---"):
                            end_idx = content.find("---", 3)
                            if end_idx != -1:
                                front_matter = content[3:end_idx].strip()
                                try:
                                    metadata = yaml.safe_load(front_matter)
                                    content = content[end_idx+3:].strip()
                                except Exception as e:
                                    logger.warning(f"Error parsing YAML front matter in {filename}: {str(e)}")
                        
                        docs.append({
                            "filename": filename,
                            "content": content,
                            "metadata": metadata
                        })
                        logger.debug(f"Loaded {filename}")
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error accessing directory {directory}: {str(e)}")
    
    logger.info(f"Loaded {len(docs)} Markdown files")
    return docs


class SentenceTransformerEmbeddings:
    """Class to create embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with the specified model."""
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized Sentence Transformer with model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding


def process_and_store(docs: List[Dict[str, Any]], db_path: str = "./chroma_db", collection_name: str = "confluence-docs", 
                      chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 100,
                      embedding_model: str = "all-MiniLM-L6-v2") -> None:
    """Process documents and store embeddings in ChromaDB.
    
    Args:
        docs: List of document dictionaries
        db_path: Path to ChromaDB database
        collection_name: Name of the collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_size: Number of embeddings to process in one batch
        embedding_model: Name of the sentence transformer model to use
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available")
        return
    
    logger.info(f"Processing {len(docs)} documents and storing embeddings")
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Initialize text splitter and embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        
        # Process documents in batches
        all_ids = []
        all_texts = []
        all_metadatas = []
        all_embeddings = []
        
        for doc in docs:
            chunks = text_splitter.split_text(doc["content"])
            logger.info(f"Split {doc['filename']} into {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{doc['filename']}-{i}"
                all_ids.append(doc_id)
                all_texts.append(chunk)
                
                # Create metadata
                metadata = {
                    "text": chunk,
                    "filename": doc["filename"],
                    "chunk_index": i
                }
                
                # Add document metadata if available
                if doc.get("metadata"):
                    for key, value in doc["metadata"].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                
                all_metadatas.append(metadata)
                
                # Process in batches
                if len(all_ids) >= batch_size:
                    logger.info(f"Creating embeddings for batch of {len(all_ids)} chunks")
                    batch_embeddings = embeddings.embed_documents(all_texts)
                    all_embeddings = batch_embeddings  # Direct assignment since we're creating all at once
                    
                    # Add to collection
                    collection.add(
                        ids=all_ids,
                        embeddings=all_embeddings,
                        metadatas=all_metadatas
                    )
                    
                    # Reset batch
                    all_ids = []
                    all_texts = []
                    all_metadatas = []
                    all_embeddings = []
        
        # Process any remaining documents
        if all_ids:
            logger.info(f"Creating embeddings for final batch of {len(all_ids)} chunks")
            batch_embeddings = embeddings.embed_documents(all_texts)
            all_embeddings = batch_embeddings  # Direct assignment
            
            # Add to collection
            collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                metadatas=all_metadatas
            )
        
        logger.info("Finished processing and storing embeddings")
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def query_collection(query: str, db_path: str = "./chroma_db", collection_name: str = "confluence-docs", 
                     n_results: int = 5, embedding_model: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """Query the ChromaDB collection.
    
    Args:
        query: Query string
        db_path: Path to ChromaDB database
        collection_name: Name of the collection
        n_results: Number of results to return
        embedding_model: Name of the sentence transformer model to use
        
    Returns:
        List of result dictionaries
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available")
        return []
    
    logger.info(f"Querying collection with: {query}")
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(name=collection_name)
        
        # Create query embedding
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        query_embedding = embeddings.embed_query(query)
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            }
            formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error querying collection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def query_ollama(query: str, context: List[Dict[str, Any]], model: str = "llama3", 
                ollama_url: str = "http://localhost:11434", 
                conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Query Ollama API with the given query and context.
    
    Args:
        query: User query
        context: List of context dictionaries
        model: Ollama model to use
        ollama_url: URL of the Ollama API
        conversation_history: Previous conversation turns
        
    Returns:
        The response from Ollama
    """
    logger.info(f"Querying Ollama API with model: {model}")
    
    try:
        # Create prompt
        prompt = f"Question: {query}\n\n"
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt += "Previous conversation:\n"
            for turn in conversation_history:
                prompt += f"User: {turn['user']}\n"
                prompt += f"Assistant: {turn['assistant']}\n\n"
            prompt += "Current question: " + query + "\n\n"
        
        prompt += "Context from Confluence:\n\n"
        
        for i, result in enumerate(context, 1):
            metadata = result["metadata"]
            title = metadata.get("title", metadata.get("filename", "Unknown"))
            prompt += f"--- Document {i}: {title} ---\n"
            prompt += metadata["text"] + "\n\n"
        
        prompt += "Based on the above context only, please answer the question concisely and accurately."
        
        # Log prompt length
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Query Ollama API
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return error_msg
        
    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


def check_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is available and running.
    
    Args:
        ollama_url: URL of the Ollama API
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


def load_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Load conversation history from a file.
    
    Args:
        session_id: Unique identifier for the conversation
        
    Returns:
        List of conversation turns
    """
    history_file = f"./conversation_history_{session_id}.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")
    return []


def save_conversation_history(session_id: str, history: List[Dict[str, str]]) -> None:
    """Save conversation history to a file.
    
    Args:
        session_id: Unique identifier for the conversation
        history: List of conversation turns
    """
    history_file = f"./conversation_history_{session_id}.json"
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")


def add_to_knowledge_base(query: str, answer: str, user_feedback: str, 
                          db_path: str = "./chroma_db", 
                          collection_name: str = "confluence-docs",
                          embedding_model: str = "all-MiniLM-L6-v2") -> bool:
    """Add user feedback to the knowledge base.
    
    Args:
        query: The original query
        answer: The system's answer
        user_feedback: The user's feedback or correction
        db_path: Path to ChromaDB database
        collection_name: Name of the collection
        embedding_model: Name of the sentence transformer model to use
        
    Returns:
        True if successful, False otherwise
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available")
        return False
    
    try:
        # Create a new document with the feedback
        content = f"Question: {query}\n\nOriginal Answer: {answer}\n\nCorrect Answer: {user_feedback}"
        
        # Create a unique ID based on timestamp
        doc_id = f"feedback-{int(time.time())}"
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(name=collection_name)
        
        # Create embedding
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        embedding = embeddings.embed_query(content)
        
        # Create metadata
        metadata = {
            "text": content,
            "filename": "user_feedback.md",
            "is_feedback": True,
            "original_query": query
        }
        
        # Add to collection
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        
        logger.info(f"Added user feedback to knowledge base with ID: {doc_id}")
        
        # Also save to a separate file for reference
        feedback_dir = "./user_feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        with open(f"{feedback_dir}/{doc_id}.txt", "w") as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding to knowledge base: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def interactive_mode(args):
    """Run the script in interactive conversational mode.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Interactive Conversational Mode ===")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'new' to start a new conversation.")
    print("Type 'feedback: your correction' to provide feedback on the last answer.")
    print("=======================================\n")
    
    # Generate a session ID based on timestamp
    session_id = str(int(time.time()))
    print(f"Session ID: {session_id}")
    
    # Load conversation history
    conversation_history = []
    last_answer = None
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Check for new conversation command
        if user_input.lower() == "new":
            session_id = str(int(time.time()))
            conversation_history = []
            print(f"\nStarting new conversation. Session ID: {session_id}")
            continue
        
        # Check for feedback
        if user_input.lower().startswith("feedback:"):
            if last_answer is None:
                print("No previous answer to provide feedback for.")
                continue
            
            feedback = user_input[9:].strip()
            if not feedback:
                print("Please provide your correction after 'feedback:'")
                continue
            
            # Get the last query from conversation history
            if conversation_history:
                last_query = conversation_history[-1]["user"]
                
                # Add to knowledge base
                success = add_to_knowledge_base(
                    last_query, last_answer, feedback,
                    args.db_path, args.collection, args.embedding_model
                )
                
                if success:
                    print("Thank you for your feedback! It has been added to the knowledge base.")
                else:
                    print("Sorry, there was an error adding your feedback.")
            else:
                print("No conversation history to provide feedback for.")
            
            continue
        
        # Process the query
        results = query_collection(
            user_input, args.db_path, args.collection, 
            args.n_results, args.embedding_model
        )
        
        if not results:
            print("No relevant information found in the Confluence data.")
            continue
        
        # Print the sources being used
        print("\nSources used:")
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            title = metadata.get("title", metadata.get("filename", "Unknown"))
            print(f"{i}. {title} (Distance: {result['distance']:.4f})")
        
        # Query Ollama with conversation history
        response = query_ollama(
            user_input, results, args.model, args.ollama_url, 
            conversation_history
        )
        
        # Print the response
        print("\nAssistant:", response)
        
        # Update conversation history
        conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # Save conversation history
        save_conversation_history(session_id, conversation_history)
        
        # Update last answer
        last_answer = response


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description='Example of using Confluence data with ChromaDB and free LLMs')
    parser.add_argument('--data-dir', default='./confluence_data/pages', 
                        help='Path to the directory containing Markdown files')
    parser.add_argument('--db-path', default='./chroma_db',
                        help='Path to ChromaDB database')
    parser.add_argument('--collection', default='confluence-docs',
                        help='Name of the ChromaDB collection')
    parser.add_argument('--query', help='The query to answer')
    parser.add_argument('--model', default='llama3', help='Ollama model to use')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                        help='URL of the Ollama API')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Sentence Transformer model to use for embeddings')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector database')
    parser.add_argument('--n-results', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive conversational mode')
    parser.add_argument('--session-id', help='Session ID for conversation history')
    parser.add_argument('--feedback', help='Provide feedback for the last answer')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available. Please install required packages.")
        return 1
    
    try:
        # Check if we need to rebuild the database
        if args.rebuild:
            logger.info("Rebuilding vector database...")
            docs = load_markdown_files(args.data_dir)
            process_and_store(docs, args.db_path, args.collection, 
                             embedding_model=args.embedding_model)
        
        # Check if Ollama is available
        if (args.query or args.interactive) and not check_ollama_available(args.ollama_url):
            logger.error(f"Ollama is not available at {args.ollama_url}. Please make sure it's running.")
            print(f"Ollama is not available at {args.ollama_url}. Please make sure it's running.")
            print("You can install Ollama from https://ollama.ai/")
            print("After installation, run: ollama pull llama3")
            return 1
        
        # Run in interactive mode if requested
        if args.interactive:
            interactive_mode(args)
            return 0
        
        # If a query is provided, search and answer
        if args.query:
            # Load conversation history if session ID is provided
            conversation_history = None
            if args.session_id:
                conversation_history = load_conversation_history(args.session_id)
            
            # Query the collection
            results = query_collection(args.query, args.db_path, args.collection, 
                                      args.n_results, args.embedding_model)
            
            if not results:
                print("No relevant information found in the Confluence data.")
                return 0
            
            # Print the sources being used
            print("\nSources used:")
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                title = metadata.get("title", metadata.get("filename", "Unknown"))
                print(f"{i}. {title} (Distance: {result['distance']:.4f})")
            
            # Query Ollama
            response = query_ollama(args.query, results, args.model, args.ollama_url, conversation_history)
            
            # Print the response
            print("\nResponse:")
            print(response)
            
            # Update conversation history if session ID is provided
            if args.session_id:
                conversation_history.append({
                    "user": args.query,
                    "assistant": response
                })
                save_conversation_history(args.session_id, conversation_history)
                print(f"\nConversation saved with session ID: {args.session_id}")
                print(f"To continue this conversation, use --session-id {args.session_id}")
            
            # Handle feedback if provided
            if args.feedback:
                success = add_to_knowledge_base(
                    args.query, response, args.feedback,
                    args.db_path, args.collection, args.embedding_model
                )
                
                if success:
                    print("\nThank you for your feedback! It has been added to the knowledge base.")
                else:
                    print("\nSorry, there was an error adding your feedback.")
        elif not args.rebuild:
            # If no query and no rebuild, show usage
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
