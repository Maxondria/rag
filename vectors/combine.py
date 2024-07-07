from typing import Any, Dict, List

from utils.short_id import generate_short_id


class VectorTextCombiner:
    """
    A utility class for combining text documents with their corresponding vector embeddings.

    This class provides a static method to combine a list of text documents and their corresponding embeddings
    into a list of dictionaries, each containing an ID, the embedding values, and metadata with the original text.

    Methods:
        combine_vector_and_text(documents: List[List[str]], embeddings: List[Any]) -> List[Dict[str, Any]]:
            Combines the documents and embeddings into a list of dictionaries with metadata.

    Example:
        documents = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        embeddings = [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ]
        result = VectorTextCombiner.combine_vector_and_text(documents, embeddings)
        # result will be:
        # [
        #     {"id": "some_id_1", "values": [1.0, 2.0, 3.0], "metadata": {"text": "a"}},
        #     {"id": "some_id_2", "values": [4.0, 5.0, 6.0], "metadata": {"text": "b"}},
        #     {"id": "some_id_3", "values": [7.0, 8.0, 9.0], "metadata": {"text": "c"}},
        #     {"id": "some_id_4", "values": [1.0, 2.0, 3.0], "metadata": {"text": "d"}},
        #     {"id": "some_id_5", "values": [4.0, 5.0, 6.0], "metadata": {"text": "e"}},
        #     {"id": "some_id_6", "values": [7.0, 8.0, 9.0], "metadata": {"text": "f"}},
        #     {"id": "some_id_7", "values": [1.0, 2.0, 3.0], "metadata": {"text": "g"}},
        #     {"id": "some_id_8", "values": [4.0, 5.0, 6.0], "metadata": {"text": "h"}},
        #     {"id": "some_id_9", "values": [7.0, 8.0, 9.0], "metadata": {"text": "i"}}
        # ]
    """
    @staticmethod
    def combine_vector_and_text(documents: List[List[str]], embeddings: List[Any]) -> List[Dict[str, Any]]:
        data_with_metadata = []

        for doc, embedding in zip(documents, embeddings):
            for text, vector in zip(doc, embedding):
                doc_id = generate_short_id(text)

                data_with_metadata.append({
                    "id": doc_id,
                    "values": vector,
                    "metadata": {"text": text},
                })

        return data_with_metadata
