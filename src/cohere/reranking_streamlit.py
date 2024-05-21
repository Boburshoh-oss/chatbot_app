import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain_cohere.llms import Cohere

# Set up the Cohere API key
os.environ["COHERE_API_KEY"] = "nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH"

# Define the QA class
class QA:
    def __init__(self, api_key):
        os.environ["COHERE_API_KEY"] = api_key
        self.llm = Cohere(temperature=0.5)
        self.compressor = CohereRerank()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def load_documents(self, file_path, file_type):
        if file_type == "text/plain":
            documents = TextLoader(file_path, encoding="utf-8").load()
        elif file_type == "application/pdf":
            documents = PyPDFLoader(file_path).load()
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            documents = Docx2txtLoader(file_path).load()
        else:
            raise ValueError("Unsupported file type. Please upload a text, PDF, or DOCX file.")
        
        texts = self.text_splitter.split_documents(documents)
        return texts

    def create_retriever(self, texts):
        retriever = FAISS.from_documents(texts, CohereEmbeddings()).as_retriever(search_kwargs={"k": 5})
        compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=retriever)
        return compression_retriever

    def answer_query(self, retriever, query):
        compressed_docs = retriever.invoke(query)
        return compressed_docs

    def generate_response(self, context, question):
        prompt_template = """
        You are an intelligent chatbot that can answer user's queries. You will be provided with relevant context based on the user's queries. 
        Your task is to analyze the user's query and generate a response for the query utilizing the context. 
        Make sure to suggest one similar follow-up question based on the context for the user to ask.

        NEVER generate a response to queries for which there is no or irrelevant context.

        Context: {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        prompt = PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        return response

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app
st.title("Document Query and Compression with Cohere and Langchain")
st.write("Enter a query to retrieve and compress documents using Cohere and Langchain.")

# Upload file
uploaded_file = st.file_uploader("Choose a text, PDF, or DOCX file", type=["txt", "pdf", "docx"])
query = st.text_input("Enter your query")

if uploaded_file and query:
    # Save the uploaded file
    file_path = f"temp.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Initialize QA object
    qa = QA(api_key="nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH")

    # Load and split documents
    texts = qa.load_documents(file_path, uploaded_file.type)

    # Create retriever
    retriever = qa.create_retriever(texts)

    # Combine current query with chat history
    combined_query = query
    if st.session_state.chat_history:
        history = "\n".join([f"Query: {chat['query']} Response: {chat['response']}" for chat in st.session_state.chat_history])
        combined_query = f"{history}\nCurrent Query: {query}"

    # Retrieve and compress documents based on combined query
    compressed_docs = qa.answer_query(retriever, combined_query)
    
    if compressed_docs:
        # Get the context from the compressed documents
        context = "\n".join([doc.page_content for doc in compressed_docs])
    else:
        context = "No relevant documents found."

    # Generate the final response using the context and the query
    final_response = qa.generate_response(context, query)

    # Add the query and final response to chat history
    st.session_state.chat_history.append({"query": query, "response": final_response})

    # Display the final response
    st.write("### Response")
    st.write(final_response)

else:
    st.write("Please upload a file and enter a query.")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Query:** {chat['query']}")
        st.write(f"**Response:** {chat['response']}")
        st.write("---")
