from langchain_cohere import ChatCohere, CohereRagRetriever
import streamlit as st
from langchain_core.messages import HumanMessage
import pdfplumber
import weaviate

API_KEY = "nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH"
client = weaviate.Client(
    "http://localhost:8080",
    additional_headers={"X-Cohere-Api-Key": f"{API_KEY}"},  # Replace with your API key
)

# Initialize the retriever with the Weaviate client
retriever = CohereRagRetriever(client)
# Cohere API kaliti
llm = ChatCohere(cohere_api_key=f"{API_KEY}")


def handle_query(query, context):
    # Foydalanuvchi kiruvchi ma'lumotni qayta ishlash
    enhanced_context = retriever.retrieve(query, context=context)
    print(enhanced_context, "nima bu")
    message = [HumanMessage(content=query, context=enhanced_context)]
    response = llm.invoke(message).content
    return response


def extract_text(file):
    # PDF fayldan matn olish
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(pages)


st.title("Cohere LLM Chatbot with File Upload")
st.write("Fayl yuklang va savollar bering!")

# Fayl yuklash
uploaded_file = st.file_uploader("PDF faylini yuklang:", type=["pdf"])
if uploaded_file is not None:
    text = extract_text(uploaded_file)
    st.write("Fayl matni muvaffaqiyatli o'qildi!")

# Savol kirish
user_query = st.text_input("Savolingizni kiriting:")
if user_query and uploaded_file:
    response = handle_query(user_query, text)
    st.write("Javob:", response)

# Chat tarixini ko'rsatish
if "history" not in st.session_state:
    st.session_state["history"] = []

if user_query:
    st.session_state["history"].append((user_query, response))
    st.write("Chat Tarixi:")
    for query, resp in st.session_state["history"]:
        st.write(f"S: {query} -> J: {resp}")
