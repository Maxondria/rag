from asyncio import TaskGroup
from typing import Any, List
from langchain_community.embeddings.ollama import OllamaEmbeddings


class Embeddings:
    """
    The Embeddings class handles the embedding of documents and queries using a specified embedding model.

    This class utilizes the OllamaEmbeddings from the langchain_community library to generate embeddings for both
    documents and queries. It provides an interface to initialize with a specific embedding model and offers
    methods to embed a list of documents or a single query.

    Attributes:
        embeddings (OllamaEmbeddings): An instance of the OllamaEmbeddings class initialized with the specified model.

    Methods:
        __init__(embedding_model: str = "llama2"):
            Initializes the Embeddings class with the specified embedding model. Default is "llama2".

        embed_documents(documents: List[str]) ->  List[List[List[float]]]:
            Generates embeddings for a list of documents and returns a list of embeddings, where each embedding is a list of floats.
            Example return:
            [
                [
                    [0.1, 0.2, 0.3, ...],  # Embedding for the first document
                    [0.4, 0.5, 0.6, ...],  # Embedding for the second document
                    ...
                ],
                ...
            ]

        query_embeddings(query: str) -> List[float]:
            Generates an embedding for a single query and returns the embedding as a list of floats.

        embedding_dimensions(embeddings: List[List[float]]) -> int:
            Returns the number of dimensions of the embeddings.
    """

    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    async def embed_documents(self, documents: List[List[str]]) -> List[Any]:
        embeddings = []
        async with TaskGroup() as taskgroup:
            for doc in documents:
                task = taskgroup.create_task(
                    self.embeddings.aembed_documents(doc))
                embeddings.append(task)
        return [task.result() for task in embeddings]

    async def _embed_query(self, query: str) -> List[float]:
        return await self.embeddings.aembed_query(query)

    @classmethod
    async def query_embeddings(cls, query: str) -> List[float]:
        """
        Returns a list of the embeddings for a given query.

        Args:
            query (str): The actual query/question

        Returns:
            list[float]: The embeddings for the given query
        """
        return await Embeddings()._embed_query(query)
