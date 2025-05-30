# Qdrant Hybrid Search with StackOverflow Dataset

This project demonstrates hybrid search (dense + sparse retrieval) using the Qdrant vector database and the StackOverflow Kaggle Questions and Answers dataset. It provides scripts to load, index, and search StackOverflow Q&A data using state-of-the-art embedding models.

## Overview

- **Qdrant**: An open-source vector database for efficient similarity search and hybrid retrieval.
- **StackOverflow Dataset**: Uses the Kaggle StackOverflow Questions and Answers CSV files as the data source.
- **Hybrid Search**: Combines dense (neural) and sparse (keyword) vector search for improved retrieval quality.

## Project Structure

- `load_stackoverflow_dafa.py`: Loads and processes StackOverflow Q&A data, creates a Qdrant collection, and uploads documents with both dense and sparse vectors.
- `hybrid_searcher.py`: Implements a class for performing hybrid search queries against the Qdrant collection using reciprocal rank fusion (RRF).
- `service.py`: (Optional) Provides a service interface (e.g., API or CLI) for querying the hybrid search functionality.
- `data/Questions.csv`, `data/Answers.csv`: Source CSV files from the StackOverflow Kaggle dataset.
- `qdrant_storage/`: Directory for Qdrant's persistent storage (used by Docker volume).

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd qdrant-hybrid-search
   ```

2. **Install Python dependencies**
   ```bash
   pip install qdrant-client sentence-transformers python-dotenv tqdm
   ```

3. **Download the StackOverflow dataset**
   - Place `Questions.csv` and `Answers.csv` from Kaggle into the `data/` directory.

4. **Create a `.env` file** in the project root with the following variables:
   ```ini
   VECTOR_STORE_URL=http://localhost:6333
   VECTOR_STORE_COLLECTION=stackoverflow
   DENSE_VECTOR_NAME=dense
   SPARSE_VECTOR_NAME=sparse
   DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   SPARSE_MODEL_NAME=prithivida/Splade_PP_en_v1
   QUESTIONS_PATH=./data/Questions.csv
   ANSWERS_PATH=./data/Answers.csv
   ROW_LIMIT=1000
   VECTOR_SEARCH_LIMIT=5
   ```
   Adjust paths and limits as needed.

5. **Start Qdrant with Docker**
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

## Usage

### 1. Load and Index Data
Run the data loader to process and upload StackOverflow Q&A pairs to Qdrant:
```bash
python load_stackoverflow_dafa.py
```

### 2. Perform Hybrid Search
You can use the `HybridSearcher` class in `hybrid_searcher.py` to perform hybrid search queries. Example usage:

```python
from hybrid_searcher import HybridSearcher
searcher = HybridSearcher(collection_name="stackoverflow")
results = searcher.search("How do I reverse a list in Python?")
for item in results:
    print(item)
```

### 3. (Optional) Run the Service
If you have implemented a service interface in `service.py`, follow its instructions to start the API or CLI for interactive querying.

## Python Module Overview

### `load_stackoverflow_dafa.py`
- Loads environment variables and configures logging.
- Initializes the Qdrant client and creates the collection if it does not exist.
- Reads questions and answers from CSV files, joining them on `ParentId`.
- For each Q&A pair, generates dense and sparse vector representations and uploads them to Qdrant with metadata.

### `hybrid_searcher.py`
- Defines the `HybridSearcher` class for querying Qdrant using hybrid search.
- Uses reciprocal rank fusion (RRF) to combine dense and sparse search results.
- Returns the top results' metadata for a given query string.

### `service.py`
- (Optional) Provides a service interface (such as a web API or CLI) for querying the hybrid search system.
- Can be extended to expose endpoints or commands for user interaction.

## Notes
- Ensure the vector names and model names in your `.env` file match those used in the scripts.
- The `ROW_LIMIT` variable can be adjusted to control how many questions are loaded for testing or production.
- The project is designed for demonstration and can be extended for larger datasets or production use.

## License

See [LICENSE](LICENSE) for details.
