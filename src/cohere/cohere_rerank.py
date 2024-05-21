# get a new token: https://dashboard.cohere.ai/

import getpass
import os

os.environ["COHERE_API_KEY"] = "nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH"


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# faiss store
from langchain_community.document_loaders import TextLoader

# from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("./data.txt", encoding="utf-8").load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, CohereEmbeddings()).as_retriever(
    search_kwargs={"k": 5}
)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = retriever.invoke(query)
# pretty_print_docs(docs)


# reranking
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_cohere.llms import Cohere


llm = Cohere(temperature=0)
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What America–second only to heart disease"
)
pretty_print_docs(compressed_docs)
