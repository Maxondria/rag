from typing import List


class TextChunker:
    """
    A utility class for chunking text into smaller segments.

    The TextChunker class is designed to split large text documents into smaller,
    more manageable chunks. This is particularly useful for processing large texts
    in natural language processing tasks where handling smaller segments can be
    more efficient and effective.

    Attributes:
        max_chunk_size (int): The maximum size of each chunk in characters. Default is 1000.

    Methods:
        chunk_text(docs: List[str]) -> List[List[str]]:
            Splits a list of documents into chunks based on the max_chunk_size.
    """

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def __chunk_text(self, text: str) -> List[str]:
        # Ensure each text ends with a double newline to correctly split paragraphs
        if not text.endswith("\n\n"):
            text += "\n\n"

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if current_chunk and (len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size):
                # if the current chunk is too big, add it to the chunks and start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""

            current_chunk += paragraph.strip() + "\n\n"

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text(self, docs: List[str]) -> List[List[str]]:
        return [self.__chunk_text(doc) for doc in docs]
