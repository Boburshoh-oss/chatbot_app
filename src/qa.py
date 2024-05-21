# qa.py
import os
import hashlib
import sqlite3
from fuzzywuzzy import fuzz
import langchain

import langchain

langchain.verbose = False

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate

# from langchain_cohere.llms import Cohere
from langchain_anthropic import AnthropicLLM, ChatAnthropic
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# model = AnthropicLLM(model='claude-2.1')


class QA:
    def __init__(self, api_key):
        os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = api_key
        self.cache = ".langchain.db"
        self._initialize_cache()
        # self.llm = Cohere(temperature=0.1)
        self.llm = ChatAnthropic(model="claude-3-opus-20240229")
        self.compressor = CohereRerank()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

    def _initialize_cache(self):
        # Initialize the SQLite database for caching
        with sqlite3.connect(self.cache) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    question TEXT,
                    value TEXT
                )
            """
            )
            conn.commit()

    def _get_from_cache(self, key):
        with sqlite3.connect(self.cache) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None

    def _set_in_cache(self, key, question, value):
        with sqlite3.connect(self.cache) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, question, value) VALUES (?, ?, ?)",
                (key, question, value),
            )
            conn.commit()

    def _set_in_cache(self, key, question, value):
        with sqlite3.connect(self.cache) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, question, value) VALUES (?, ?, ?)",
                (key, question, value),
            )
            conn.commit()

    def load_documents(self, file_path, file_type):
        if file_type == "text/plain":
            documents = TextLoader(file_path, encoding="utf-8").load()
        elif file_type == "application/pdf":
            documents = PyPDFLoader(file_path).load()
        elif (
            file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            documents = Docx2txtLoader(file_path).load()
        else:
            raise ValueError(
                "Unsupported file type. Please upload a text, PDF, or DOCX file."
            )

        texts = self.text_splitter.split_documents(documents)
        return texts

    def create_retriever(self, texts):
        retriever = FAISS.from_documents(texts, CohereEmbeddings()).as_retriever(
            search_kwargs={"k": 5}
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=retriever
        )
        return compression_retriever

    def answer_query(self, retriever, query):
        compressed_docs = retriever.invoke(query)
        return compressed_docs

    def generate_response(self, context, question):
        normalized_question = self._normalize_query(question)
        cache_key = self._generate_cache_key(normalized_question)
        cached_response = self._get_similar_cached_response(normalized_question)

        if cached_response:
            print("Cache hit")
            return cached_response

        print("Cache miss")
        prompt_template = """
        You are an intelligent chatbot that can answer user's queries. You will be provided with relevant context based on the user's queries. 
        Your task is to analyze the user's query and generate a response for the query utilizing the context. 
        Make sure to suggest one similar follow-up question based on the context for the user to ask.

        NEVER generate a response to queries for which there is no or irrelevant context.
        
        Previous conversation:
        {chat_history}

        Context: {context}

        Question: {question}

        Answer:
        """
        # PROMPT = PromptTemplate(
        #     template=prompt_template,
        #     input_variables=["chat_history", "context", "question"],
        # )
        memory_variables = self.memory.load_memory_variables({"chat_history": ""})
        # print(memory_variables)
        chat_history = memory_variables.get("chat_history", "")
        # prompt = PROMPT.format(
        #     chat_history=chat_history, context=context, question=question
        # )
        print(chat_history,"history bormi")
        PROMPT = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are an intelligent chatbot that can answer user's queries. You will be provided with relevant context based on the user's queries. Your task is to analyze the user's query and generate a response for the query utilizing the context. Make sure to suggest one similar follow-up question based on the context for the user to ask. NEVER generate a response to queries for which there is no or irrelevant context."
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ],
        )
        promt = PROMPT.format(context=context, question=question,chat_history=chat_history)
        print(promt)
        response = self.llm.invoke(promt).content
        # Cache the response
        self._set_in_cache(cache_key, normalized_question, response)

        # Store the conversation in memory
        self.memory.chat_memory.add_message({"role": "user", "content": question})
        self.memory.chat_memory.add_message({"role": "assistant", "content": response})

        return response

    def _normalize_query(self, query):
        # Normalize the query by lowercasing and stripping whitespace
        return query.lower().strip()

    def _generate_cache_key(self, question):
        return hashlib.md5(question.encode()).hexdigest()

    def _get_similar_cached_response(self, question, threshold=80):
        normalized_question = self._normalize_query(question)
        print(normalized_question, "normalized question")
        with sqlite3.connect(self.cache) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT question, value FROM cache")
            rows = cursor.fetchall()
            for cached_question, value in rows:
                print(cached_question, "cached question")
                if fuzz.ratio(normalized_question, cached_question) > threshold:
                    return value
        return None
