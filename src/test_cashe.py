# test_cache.py
import os
import time
from qa import QA
from langchain_community.cache import SQLiteCache, InMemoryCache
from langchain.globals import set_llm_cache


# Set up the Cohere API key
os.environ["COHERE_API_KEY"] = "nAcNQjlxT9dYt4pfP9MLYiRG0pNiBvCjT2yZZ0yH"

# Define the question and context for testing
question = "Tell me a joke"
context = "Here is some relevant context for the joke."

def measure_time_and_response(qa, question, context):
    start_time = time.time()
    response = qa.generate_response(context, question)
    elapsed_time = time.time() - start_time
    return elapsed_time, response

# Test with In-Memory Cache
print("Testing with In-Memory Cache")
set_llm_cache(InMemoryCache())

qa = QA(api_key=os.getenv("COHERE_API_KEY"))

# Measure the time for the first query
elapsed_time, response = measure_time_and_response(qa, question, context)
print(f"First query (In-Memory): {elapsed_time:.2f} ms, Response: {response}")

# Measure the time for the second query (should be faster)
elapsed_time, response = measure_time_and_response(qa, question, context)
print(f"Second query (In-Memory): {elapsed_time:.2f} ms, Response: {response}")

# Test with SQLite Cache
print("Testing with SQLite Cache")
# os.remove(".langchain.db") if os.path.exists(".langchain.db") else None
set_llm_cache(SQLiteCache(database_path=".langchain2.db"))

qa = QA(api_key=os.getenv("COHERE_API_KEY"))

# Measure the time for the first query
elapsed_time, response = measure_time_and_response(qa, question, context)
print(f"birinchi query (SQLite): {elapsed_time:.2f} ms, Response: {response}")

# Measure the time for the second query (should be faster)
elapsed_time, response = measure_time_and_response(qa, question, context)
print(f"ikkinchi query (SQLite): {elapsed_time:.2f} ms, Response: {response}")
