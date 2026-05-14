import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("📺 YouTube Video Q&A Assistant")

# api_key = "aavvAIzaSyBkaIy6DOUyZ1puRpEvvVcMyKcyOSRBPBA"
api_key = st.secrets["GOOGLE_API_KEY"]

def get_video_id(url):
    """Extracts the video ID from various YouTube URL formats."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

with st.sidebar:
    st.header("Setup")
    video_url = st.text_input("Enter YouTube Video URL:")
    process_button = st.button("Process Video")

if process_button and video_url:
    video_id = get_video_id(video_url)

    if video_id:
        with st.spinner("Fetching transcript and building index..."):
            try:
                # Extraction
                ytt = YouTubeTranscriptApi()
                transcript_obj = ytt.fetch(video_id=video_id, languages=['en'])
                final_transcript = " ".join([s.text for s in transcript_obj])

                # Document Chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = splitter.create_documents(texts=[final_transcript])

                # Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key
                )
                vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)

                # session saving
                st.session_state.vector_store = vector_store
                st.session_state.video_processed = True
                st.success("Video processed successfully!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Invalid YouTube URL.")

# QA
st.divider()

if "video_processed" in st.session_state:
    user_question = st.text_input("Ask a question about the video content:")

    if user_question:
        with st.spinner("Generating answer..."):
            # Retrieval
            retriever = st.session_state.vector_store.as_retriever()
            results = retriever.invoke(user_question)
            context_text = "\n\n".join([doc.page_content for doc in results])

            # Generation
            llm = ChatGoogleGenerativeAI(
                model='models/gemini-flash-latest',
                google_api_key=api_key
            )

            template = PromptTemplate(template="""
            You are a helpful QA bot who gives indept answers. Answer the question using ONLY the provided transcript context.
            Provide a concise, indepth answer. include conversational filler, introductory phrases.

            Very Important: If the context is insufficient to answer the question, output exactly and only: "It is not mentioned in the provided context."

            Context: {context}
            Question: {question}
            Answer:
            """, input_variables=['context', 'question'])

            prompt = template.format(context=context_text, question=user_question)
            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                # trying to only print the text in response.content
                text_only = "".join(block.get("text", "") for block in response.content if block.get("type") == "text")
                st.write(text_only)
            else:
                st.write(response.content)
else:
    st.info("Please enter a YouTube URL and click 'Process Video' to start.")
