import re, time, numpy as np
from difflib import SequenceMatcher
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import nltk
import os
import json
from difflib import SequenceMatcher
import streamlit as st
from dotenv import load_dotenv
import faiss

nltk.download('punkt')

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def fetch_transcript(videoid, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return YouTubeTranscriptApi.get_transcript(videoid)
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception:
            time.sleep(delay)
    return None

def tokenize_and_chunk(transcript, max_tokens=100):
    full_text = ' '.join([entry['text'] for entry in transcript])
    sentences = nltk.sent_tokenize(full_text)
    chunks = []
    current_chunk = []
    token_count = 0

    for s in sentences:
        t = len(s.split())
        if token_count + t > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk, token_count = [s], t
        else:
            current_chunk.append(s)
            token_count += t

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def find_start_time(chunk, transcript, window=10):
    chunk_start = ' '.join(chunk.split()[:window]).lower()
    best_match = None
    best_ratio = 0.0

    for entry in transcript:
        entry_start = ' '.join(entry['text'].split()[:window]).lower()
        ratio = SequenceMatcher(None, chunk_start, entry_start).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = entry['start']
    
    return best_match


def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=[text.replace("\n", " ")], model=model)
        return response.data[0].embedding
    except:
        return None
    
st.set_page_config(page_title="🎥 Multimodal Video Chat", layout="centered")
st.title("🎬 Ask Questions About Any YouTube Video")

# --- Input Section ---
video_url = st.text_input("📎 Paste YouTube Video Link")

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.markdown("#### ▶️ Video Preview")
        st.components.v1.html(
            f"""
            <iframe width="100%" height="360" 
            src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" allowfullscreen></iframe>
            """,
            height=380,
        )
    else:
        st.warning("⚠️ Could not detect a valid YouTube video ID.")

query = st.text_input("💬 Ask a question about the video")

# --- Trigger Button ---
if st.button("Search"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    videoid = extract_video_id(video_url)
    if not videoid:
        st.warning("Invalid YouTube URL.")
        st.stop()

    transcript = fetch_transcript(videoid)
    if not transcript:
        st.error("⚠️ Could not fetch transcript for this video.")
        st.stop()

    st.info("Processing transcript...")

    chunks = tokenize_and_chunk(transcript)
    final_chunks = []
    for chunk in chunks:
        start_time = find_start_time(chunk, transcript)
        emb = get_embedding(chunk)
        if emb:
            final_chunks.append({"start": start_time, "text": chunk, "embedding": emb})
        time.sleep(0.2)

    # Build FAISS index
    vectors = np.array([c['embedding'] for c in final_chunks]).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1]) #L2 = Euclidean distance
    index.add(vectors)

    # Embed query
    q_emb = get_embedding(query)
    if not q_emb:
        st.error("Failed to embed query.")
        st.stop()

    q_vector = np.array([q_emb]).astype('float32')
    _, indices = index.search(q_vector, 3)

    st.success("Done! Here are the most relevant moments:")

    # Start by getting a list of result chunks
    results = [final_chunks[i] for i in indices[0]]

    # Then display each result in Streamlit
    for result in results:
        ts = int(result['start'] // 5 * 5)
        link = f"{video_url}&t={ts}"
        st.markdown(f"**🕒 [Jump to {ts}s]({link})**")
        st.markdown(f"> {result['text']}")
        st.markdown("---")
