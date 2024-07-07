import os
from typing import Any, Dict, List
from pinecone import Pinecone, QueryResponse


class PineconeOps:
    """
    A class to handle upserting data with metadata into a Pinecone index.

    This class provides methods to initialize a connection to a specified Pinecone index and upsert data with metadata into it.

    Attributes:
        pc (Pinecone): An instance of the Pinecone class initialized with the API key.
        index_name (str): The name of the Pinecone index.
        index (Pinecone.Index): The Pinecone index instance.

    Methods:
        __init__(index_name: str):
            Initializes the PineconeDataUpserter with the given index name.

        upsert_data(data_with_metadata: list[dict[str, Any]]) -> None:
            Upserts data with metadata into the Pinecone index.
    """
    """
    A class to handle upserting data with metadata into a Pinecone index.
    """

    def __init__(self, index_name: str):
        """
        Initializes the PineconeDataUpserter with the given API key and index name.

        Args:
        - api_key (str): The API key for Pinecone.
        - index_name (str): The name of the Pinecone index.
        """
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.index = self.pc.Index(index_name)

    def upsert_data(self, data_with_metadata: list[dict[str, Any]]) -> None:
        """
        Upsert data with metadata into the Pinecone index.

        Args:
        - data_with_metadata (List[Dict[str, Any]]): A list of dictionaries, each containing data with metadata.

        Returns:
        - None
        """
        self.index.upsert(vectors=data_with_metadata)

    def query_index(self, query_embeddings: List[float], top_k: int = 10, include_metadata: bool = True) -> QueryResponse:
        """
        Query data from the Pinecone index.
        """
        return self.index.query(vector=query_embeddings,
                                top_k=top_k, include_metadata=include_metadata)
