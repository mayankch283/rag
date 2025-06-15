import pickle
import os
from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
import requests
import torch

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_or_create_chunks(pdf_path):
    # Create cache filename based on PDF name
    cache_dir = "../cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}.pickle")
    
    # If cache exists, load from it
    if os.path.exists(cache_file):
        print("Loading chunks from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # If no cache, create chunks and save them
    print("Creating new chunks...")
    text_splitter = SemanticChunker(model)
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    
    # Save chunks to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    return chunks

# Use the caching function
chunks = get_or_create_chunks("../BERTforIR.pdf")

query = input("Enter your query: ")

chunk_texts = [chunk.page_content for chunk in chunks]
print("Number of chunks:", len(chunk_texts))

embed_chunks = model.embed_documents(chunk_texts)
embed_query = model.embed_query(query)

# Calculate cosine similarity
sim = util.cos_sim(embed_query, embed_chunks)

k = 5
top_results = torch.topk(sim, k=k)
print(top_results.indices[0])
context = "".join([chunk_texts[i] for i in top_results.indices[0]])

print(context)

#ai/smollm2 on DMR
# response = requests.post(
#     "http://localhost:12434/engines/llama.cpp/v1/chat/completions",
#     headers={"Content-Type": "application/json"},
#     json={
#         "model": "ai/smollm2",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant. Answer questions based on the following context: " + context
#             },
#             {
#                 "role": "user",
#                 "content": query
#             }
#         ]
#     }
# )

# print("Response:", response.json()['choices'][0]['message']['content'])

