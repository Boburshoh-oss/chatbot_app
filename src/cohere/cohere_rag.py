from pprint import pprint

from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

user_query = "When was Cohere started?"

# Create Cohere's chat model and embeddings objects
llm = ChatCohere()
cohere_embeddings = CohereEmbeddings()

# Load text files and split into chunks, you can also use data gathered elsewhere in your application
raw_documents = TextLoader("test.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
# Create a vector store from the documents
db = Chroma.from_documents(documents, cohere_embeddings)
input_docs = db.as_retriever().get_relevant_documents(user_query)

rag = CohereRagRetriever(llm=llm)
docs = rag.get_relevant_documents(
    user_query,
    documents=input_docs,
)

answer = docs.pop()

pprint("Relevant documents:")
pprint(docs)

pprint("Answer:")
pprint(answer.page_content)
pprint(answer.metadata["citations"])
