import asyncio

from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from db.vector_db import PineconeOps
from embeds.embeddings import Embeddings

from readers.pdf import PDFReader
from splitters.chunk import TextChunker
from vectors.combine import VectorTextCombiner


load_dotenv()


LLM = Ollama(model="llama2")


async def main():
    docs = PDFReader(directory="docs").load()
    chunks = TextChunker().chunk_text(docs)

    doc_embeddings = await Embeddings().embed_documents(documents=chunks)

    data_with_meta_data = VectorTextCombiner.combine_vector_and_text(
        documents=chunks, embeddings=doc_embeddings)

    vector_db = PineconeOps(index_name="a-look-at-rag")
    vector_db.upsert_data(data_with_meta_data)

    user_query = input("Please enter your query: ")

    query_embeddings = await Embeddings.query_embeddings(
        query=user_query) if user_query else None

    query_response = vector_db.query_index(
        query_embeddings=query_embeddings) if query_embeddings else None

    if query_response:
        context = " ".join(
            [result["metadata"]["text"] for result in query_response.matches])

        prompt = f"{context} \n {user_query}"

        response = LLM.invoke(input=prompt)

        print(response)


if __name__ == "__main__":
    asyncio.run(main())
