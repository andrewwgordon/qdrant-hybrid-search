"""
service.py

Provides a FastAPI web service for performing hybrid search queries against a Qdrant vector database collection.
"""

from fastapi import FastAPI
from os import environ
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# The file where HybridSearcher is stored
from hybrid_searcher import HybridSearcher

app = FastAPI()

# Create a neural searcher instance
hybrid_searcher = HybridSearcher(collection_name=environ["VECTOR_STORE_COLLECTION"])


@app.get("/api/search")
def search_startup(q: str):
    """
    Perform a hybrid search for the given query string.

    Args:
        q (str): The query string to search for.

    Returns:
        dict: A dictionary containing the search results.
    """
    return {"result": hybrid_searcher.search(text=q)}


if __name__ == "__main__":
    """
    Run the FastAPI application using Uvicorn when executed as a script.
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
