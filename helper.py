import os
from dotenv import load_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize the Google AI embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the Gemini language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)


def create_db_from_youtube_url(video_url: str) -> FAISS:
    """
    Creates a FAISS vector database from a YouTube video transcript.
    
    Args:
        video_url: The URL of the YouTube video.

    Returns:
        A FAISS database object containing the video's content.
    """
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

# print(create_db_from_youtube_url("https://youtu.be/P5BRLfA3fGs?si=U0yGNmT0TqDWYeAh"))

