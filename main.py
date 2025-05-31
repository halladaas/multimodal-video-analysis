import re
import nltk
import os
import openai
import json
import time
from difflib import SequenceMatcher
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import faiss
from youtube_transcript_api import YouTubeTranscriptApi

st.set_page_config(page_title="Video Chat RAG", layout="wide")

nltk.download('punkt')
url = "https://www.youtube.com/watch?v=iHpQjUyaBRQ&ab_channel=anythinggoeswithemmachamberlain"

def extract_video_id(_url):
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, _url)
    return match.group(1) if match else None
        
videoid = extract_video_id(url)

try:
    transcript = YouTubeTranscriptApi.get_transcript(videoid)
except Exception as e:
    st.error(f"⚠️ Could not fetch transcript: {e}")
    st.stop()
    
#--------------------Time-based chunking of the transcript (30s)----------------------
chunks_duration = 30
chunks = {}
for entry in transcript:
    start = float(entry['start'] // chunks_duration) * chunks_duration
    if start not in chunks:
        chunks[start] = []
    chunks[start].append(entry['text'])
    
#convert to structured output
chunked_transcript = []
for start, texts in chunks.items():
    merged_text = ' '.join(texts)
    chunked_transcript.append({
        'start' : start,
        'text': merged_text
    })

'''
#print
for chunk in chunked_transcript[:5]:
    print(f"Start: {chunk['start']}\nText: {chunk['text']}\n") 
'''
#--------------------Token-based chunking of the transcript----------------------
# Ensure nltk punkt tokenizer is downloaded

# Tokenize the full text into sentences
full_text = ' '.join(chunk['text'] for chunk in chunked_transcript)
sentences = nltk.sent_tokenize(full_text)

# Clean sentences to remove any leading/trailing whitespace
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

#  Group Sentences into Token-Limited Chunks (~100 tokens each)
token_chunks = []
current_chunk = []
current_token_count = 0
max_tokens = 100

for sentence in sentences:
    
    token_count = len(sentence.split())
    
    if current_token_count + token_count > max_tokens:
        token_chunks.append(' '.join(current_chunk))
        current_chunk = [sentence]
        current_token_count = token_count
    else:
        current_chunk.append(sentence)
        current_token_count += token_count
        
# If there's any remaining chunk after the loop, add it    
if current_chunk:
    token_chunks.append(' '.join(current_chunk))



#Get the timestamp for each token chunk
def find_start_time(chunk, transcriprt, window=10):
    
    """
    Find the start time of a chunk in the transcript.
    Uses a sliding window to find the best match.
    """
    best_match = None
    best_ratio = 0.0
    
    for entry in transcriprt:
        text = entry['text']
        start = entry['start']
        
        # Use SequenceMatcher to find the best match
        ratio = SequenceMatcher(None, chunk, text).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = start
            
    return best_match

    
final_chunks = []

for chunk in token_chunks:
    start_time = find_start_time(chunk, transcript)
    final_chunks.append({
        'start': start_time,
        'text': chunk
    })

# # Example print
# print("Final chunks (first 3):\n")
# for chunk in final_chunks[:3]:
#     print(f"Start: {chunk['start']}s\nText: {chunk['text']}...\n")

#--------------------OpenAI API Key Setup----------------------
# Ensure you have set the OPENAI_API_KEY environment variable in your shell

# Get the environment variable from your shell
openai.api_key = os.getenv("OPENAI_API_KEY")

# Optional test
print("Key loaded:", openai.api_key[:6] + "..." if openai.api_key else "Not found")

#--------------------Embed the chunks with OpenAI API----------------------

def get_embedding(text, model="text-embedding-3-small"):
    # Ensure text is not too long (model limit is ~8191 tokens)
    text = text.replace("\n", " ")  # optional cleanup
    try:
        response = openai.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Failed to embed: {e}")
        print("Text that failed:", text[:100], "...")
        return None
    
for chunk in final_chunks:
    embedding = get_embedding(chunk['text'])
    chunk['embedding'] = embedding
    time.sleep(0.2)  # polite rate-limiting
    
# save embeddings to a file
output_file = "embeddings.json"
with open(output_file, 'w') as f:
    json.dump(final_chunks, f)

#--------------------Load embeddings and create FAISS index----------------------
# Get the embeddings
embedding_vectors = np.array([chunk['embedding'] for chunk in final_chunks]).astype('float32')

# Create the FAISS index
index = faiss.IndexFlatL2(embedding_vectors.shape[1])  # L2 = Euclidean distance
index.add(embedding_vectors)

def search_chunks(query, k=3):
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(final_chunks[i])
    
    return results

'''
# Example search
query = "what is the main topic of the video?"
results = search_chunks(query)

for res in results:
    print(f"Time: {res['start']}s\nText: {res['text']}...\n")
'''

# --- Streamlit App UI ---
st.title("🎥 Multimodal Video Chat")

st.markdown("Paste your video link, then ask a question about the video content.")

video_url = st.text_input("🔗 YouTube URL (for timestamp links only)")
query = st.text_input("💬 Ask a question about the video")

if st.button("Search"):
    if not query:
        st.warning("Please enter a question.")
    else:
        results = search_chunks(query)

        if results:
            for res in results:
                timestamp = int(res['start'])
                timestamp_link = f"{video_url}&t={timestamp}" if video_url else f"(Time: {timestamp}s)"
                st.markdown(f"**🕒 [Jump to {timestamp}s]({timestamp_link})**")
                st.markdown(f"> {res['text']}")
        else:
            st.warning("No results found.")
