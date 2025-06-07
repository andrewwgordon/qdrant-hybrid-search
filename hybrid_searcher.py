"""
Module: hybrid_searcher

Provides the HybridSearcher class for performing hybrid (dense + sparse) search queries
against a Qdrant vector database collection using reciprocal rank fusion (RRF).
"""
from qdrant_client import QdrantClient, models
import dotenv
import logging
from os import environ

# Load environment variables from .env file
dotenv.load_dotenv()
# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class HybridSearcher:
    """
    HybridSearcher enables hybrid search (dense + sparse) over a Qdrant collection.

    Attributes:
        collection_name (str): The name of the Qdrant collection to search.
        qdrant_client (QdrantClient): The Qdrant client instance.
    """

    def __init__(self, collection_name):
        """
        Initialize the HybridSearcher with a collection name.

        Args:
            collection_name (str): The name of the Qdrant collection to search.
        """
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=environ["VECTOR_STORE_URL"])

    def search(self, text: str):
        """
        Perform a hybrid search using reciprocal rank fusion (RRF) over dense and sparse vectors.

        Args:
            text (str): The query text to search for.

        Returns:
            list: A list of payloads (metadata) from the search results, filtered by minimum score.
        """
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # Use reciprocal rank fusion for combining results
            ),
            prefetch=[
                # Prefetch dense vector search
                models.Prefetch(
                    query=models.Document(text=text, model=environ["DENSE_MODEL_NAME"]),
                    using=environ["DENSE_VECTOR_NAME"],
                ),
                # Prefetch sparse vector search
                models.Prefetch(
                    query=models.Document(
                        text=text, model=environ["SPARSE_MODEL_NAME"]
                    ),
                    using=environ["SPARSE_VECTOR_NAME"],
                ),
            ],
            query_filter=None,  # No additional filters applied
            limit=int(environ["VECTOR_SEARCH_LIMIT"]),  # Limit the number of results
        ).points
        # Extract and return the payload (metadata) from each result point, filtering by min score
        metadata = [
            point.payload
            for point in search_result
            if point.score >= float(environ["VECTOR_STORE_SEARCH_MIN_SCORE"])
        ]
        return metadata
