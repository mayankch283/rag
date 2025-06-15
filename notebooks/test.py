from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import requests
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

loader = PyPDFLoader("../moose-mountain.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

query = input("Enter your query: ")

chunk_texts = [chunk.page_content for chunk in chunks]

q_embeddings = model.encode(query, convert_to_tensor=True)
embeddings = model.encode(chunk_texts, convert_to_tensor=True)

sim = util.euclidean_sim(q_embeddings, embeddings)[0]
k = 3
top_results = torch.topk(sim, k=k)

context = ''.join([chunk_texts[idx] for idx in top_results.indices])

print(context)

#ai/smollm2 on DMR
response = requests.post(
    "http://localhost:12434/engines/llama.cpp/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "ai/smollm2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based on the following context: " + context
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
)

print("Response:", response.json()['choices'][0]['message']['content'])
    
