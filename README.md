# Multimodal Video Chat – YouTube AI Assistant

This project lets users **chat with any YouTube video** using natural language. It extracts video transcripts, generates semantic embeddings, and finds the most relevant timestamped video moments based on user questions.

```markdown
├── app.py                # preprocessing script and Streamlit app UI  
├── embeddings.json  # Saved transcript chunks + embeddings  
├── requirements.txt  
└── README.md
```
---

## Features
- 🧠 Natural language question answering on video content
- 🕒 Timestamped chunk navigation and deep linking
- 🎞️ YouTube video preview inside the app
- ⚡ Powered by OpenAI embeddings + FAISS similarity search
- 💻 Built with Streamlit for a fast and clean UI

---

## How It Works
1. User pastes a YouTube video link  
2. The app fetches the transcript using `youtube-transcript-api`  
3. It splits the transcript into meaningful chunks using tokenizations 
4. Each chunk is embedded using OpenAI's `text-embedding-3-small`  
5. A FAISS index enables fast semantic similarity search  
6. Users can ask a question and see relevant parts of the video

---

## Technologies
- Python  
- OpenAI API  
- FAISS  
- Streamlit  
- YouTube Transcript API  
- NLTK

---

## Setup

1. Clone this repo  
2. Create a `.env` file with your OpenAI API key: OPENAI_API_KEY=your-api-key-here
3. Install dependencies:
``` bash
pip install -r requirements.txt
```
4. Run the app:
``` bash
streamlit run app.py
```
 
