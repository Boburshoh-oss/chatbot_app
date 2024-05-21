import os
import streamlit as st
from qa import QA

# Set up the Cohere API key
os.environ["ANTHROPIC_API_KEY"] = "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"

# Initialize session state for chat history and QA object if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa" not in st.session_state:
    st.session_state.qa = QA(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

    # Use the QA object from session state
    qa = st.session_state.qa

    # Load and split documents
    texts = qa.load_documents(file_path, uploaded_file.type)

    # Create retriever
    retriever = qa.create_retriever(texts)

    # Combine current query with chat history
    combined_query = query
    # if st.session_state.chat_history:
    #     history = "\n".join(
    #         [
    #             f"Query: {chat['query']} Response: {chat['response']}"
    #             for chat in st.session_state.chat_history
    #         ]
    #     )
    #     combined_query = f"{history}\nCurrent Query: {query}"

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
