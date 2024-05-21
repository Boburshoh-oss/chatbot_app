import streamlit as st
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

API_KEY = "nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH" # Use a secure way to handle the API key

# Initialize Cohere's chat model, embeddings, and text processing
llm = ChatCohere(cohere_api_key=API_KEY)
cohere_embeddings = CohereEmbeddings(cohere_api_key=API_KEY)

# Streamlit User Interface
st.title("Cohere Document Search and Chatbot")

# File uploader and document processing
if 'documents' not in st.session_state or st.button("Upload New Text"):
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"], key="uploader")
    if uploaded_file:
        text = uploaded_file.getvalue().decode("utf-8")

        def generate_tokens(s):
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_text(s)
            return text_splitter.create_documents(splits)
        print(generate_tokens(text))
        st.session_state['documents'] = generate_tokens(text)
        db = Chroma.from_documents(st.session_state['documents'], cohere_embeddings)
        st.session_state['db'] = db.as_retriever()

if 'db' in st.session_state:
    user_query = st.text_input("Enter your query:")
    if user_query:
        input_docs = st.session_state['db'].get_relevant_documents(user_query)
        rag = CohereRagRetriever(llm=llm)
        docs = rag.get_relevant_documents(user_query, documents=input_docs)

        if docs:
            st.write("Relevant documents:")
            for doc in docs[:-1]:  # Display all but the last document as relevant
                st.text(doc)
            st.write("Answer from the most relevant document:")
            st.text(docs[-1])  # Display the last document as the answer
        else:
            st.write("No relevant documents found.")
