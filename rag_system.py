import os
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

with open("my_knowledge.txt") as f:
    knowledge_text = f.read()

print("=== Document loaded ===\n")

# split into small chunks so the model doesn't choke on a wall of text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,  # a bit of overlap so we don't cut context in half
    length_function=len
)

chunks = text_splitter.split_text(knowledge_text)

print(f"=== {len(chunks)} chunks created ===")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {chunk[:80]}...")

print("\n=== Loading embedding model ===")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embed_model.encode(chunks)

print(f"Embedding shape: {chunk_embeddings.shape}")

# throw the vectors into faiss so we can do similarity search later
d = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(chunk_embeddings).astype('float32'))

print(f"\n=== FAISS index ready — {index.ntotal} vectors stored ===")

print("\n=== Loading generator model ===")
generator = pipeline('question-answering', model='deepset/roberta-base-squad2')


def answer_question(query: str, k: int = 2) -> str:
    query_embedding = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    best_distance = distances[0][0]
    print(f"Best retrieval distance: {best_distance:.4f}")

    # if the nearest chunk is still pretty far away, nothing relevant was found
    if best_distance > 1.1:
        print(f"\nQuery   : {query}")
        print(f"Answer  : I don't have that information.")
        print("-" * 60)
        return "I don't have that information."

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    result = generator(question=query, context=context)
    answer = result['answer'] if result['score'] > 0.001 else "I don't have that information."

    print(f"\nQuery   : {query}")
    print(f"Context :\n{context}\n")
    print(f"Score   : {result['score']:.4f}")
    print(f"Answer  : {answer}")
    print("-" * 60)
    return answer


if __name__ == "__main__":
    answer_question("What is the WFH policy?")
    answer_question("What is the company's dental plan?")
    answer_question("What programming languages does the company use?")