# qdrant-hybrid-search

Hybrid Search Example in Qdrant

This project demonstrates how to perform hybrid search (dense + sparse retrieval) using Qdrant as a vector database. It includes code for loading data, performing hybrid search queries, and running a service interface. The solution is designed to work with a Qdrant instance running as a Docker service.

## Prerequisites

- Python 3.8+
- Qdrant server running (recommended: via Docker)
- Required Python packages (see below)

### Running Qdrant with Docker

You can start Qdrant using Docker with:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

This will expose Qdrant on `localhost:6333` and persist data in the `qdrant_storage` directory.

## Python Files Overview

### `data_load.py`
Responsible for loading and uploading data into the Qdrant collection