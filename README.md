Building sentence encoder for effective RAG for LLMs - 

RAG requires careful consideration of chunking strategies and embedding model selection to optimize retrieval performance.
The process involves implementing techniques like query rewriting and embedding transformation while ensuring proper alignment between the retriever and LLM components.
Key aspects include experimenting with different block sizes, potentially fine-tuning embedding models for specialized domains, and implementing post-retrieval processing to enhance the quality of retrieved results.

Biggest Current Challenges:
    Embedding meaning in context, cross-modality, interpretability, and scalability for long texts or agent memory.

Helpful links:

https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
https://medium.aiplanet.com/advanced-retrieval-and-evaluation-hybrid-search-with-minicoil-using-qdrant-and-langgraph-6fbe5e514078
https://github.com/UKPLab/sentence-transformers
https://qdrant.tech/articles/
https://www.pinecone.io/learn/series/faiss/