import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.genai as genai  

DATASET_FILE = "schemes_dataset.csv"
CHAT_HISTORY_FILE = "chat_history.json"
FAISS_INDEX_FILE = "schemes_faiss.index"
METADATA_FILE = "schemes_metadata.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
GOOGLE_API_KEY = "AIzaSyBvejIwuCsivIYoi4jUwImQsvvoWWM9Pyo"


try:
    df = pd.read_csv(DATASET_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_FILE, encoding="latin1")

df.fillna("", inplace=True)
df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
if "scheme_name" not in df.columns:
    df["scheme_name"] = [f"scheme_{i}" for i in range(len(df))]


if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
else:
    chat_history = []

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2, ensure_ascii=False)


embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts):
    vectors = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype("float32")


def build_faiss_index():
    texts = df["combined_text"].tolist()
    vectors = embed_texts(texts)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    metadata = []
    for i, row in df.iterrows():
        metadata.append({
            "index": i,
            "scheme_name": row.get("scheme_name", ""),
            "objective": row.get("objective", ""),
            "benefit": row.get("benefit", ""),
            "eligibility": row.get("eligibility", ""),
            "documents_required": row.get("documents_required", ""),
            "combined_text": row.get("combined_text", "")
        })
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[+] Built FAISS index with {len(metadata)} entries.")

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        build_faiss_index()
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_faiss_index()


def retrieve(query, top_k=TOP_K):
    q_vector = embed_texts([query])
    faiss.normalize_L2(q_vector)
    D, I = index.search(q_vector, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        hits.append({"score": float(score), "meta": metadata[idx]})
    return hits

def build_context_from_hits(hits):
    if not hits:
        return "No matching government schemes found in the dataset."
    context_parts = []
    for h in hits:
        m = h["meta"]
        context_parts.append(
            f"Scheme: {m['scheme_name']}\n"
            f"Objective: {m['objective']}\n"
            f"Benefit: {m['benefit']}\n"
            f"Eligibility: {m['eligibility']}\n"
            f"Documents: {m['documents_required']}\n"
        )
    return "\n\n".join(context_parts)


def generate_response(prompt):
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print("[!] Generation error:", e)
        return "Sorry, I cannot respond right now. Check your API key or network."


def answer_query(user_input):
    scheme_keywords = ["scheme", "government", "benefit", "eligibility",
                       "documents", "objective", "education", "student",
                       "financial assistance"]

    if any(word in user_input.lower() for word in scheme_keywords):
        hits = retrieve(user_input, TOP_K)
        context = build_context_from_hits(hits)
        prompt = f"""
You are a helpful assistant for government schemes.
User query: {user_input}

Relevant schemes retrieved (from dataset):
{context}

Answer in simple, conversational terms, referencing the relevant schemes above.
"""
    else:
        conversation_history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" 
                                          for h in chat_history[-10:]])
        prompt = f"""
You are a helpful assistant.
Conversation so far:
{conversation_history}

User query: {user_input}
Answer naturally and helpfully.
"""
    response = generate_response(prompt)
    chat_history.append({"user": user_input, "bot": response})
    save_chat_history()
    return response


def run_chat():
    print("💬 RAG Chatbot started! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        reply = answer_query(user_input)
        print("\n🤖 Bot:", reply)


if __name__ == "__main__":
    run_chat()
