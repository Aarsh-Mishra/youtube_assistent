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

def get_response_from_query(db: FAISS, query: str, k=4) -> str:
    """
    Queries the FAISS database and generates a response using Gemini.
    
    Args:
        db: The FAISS database object.
        query: The user's question.
        k: The number of relevant documents to retrieve.

    Returns:
        The AI-generated response as a string.
    """
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful YouTube assistant. Your task is to answer questions about a video 
        based on its provided transcript.

        Answer the following question: {question}
        
        Use only the following transcript excerpt to answer: {docs}

        If the transcript excerpt doesn't contain the answer, state "I don't have enough information to answer."
        Your answer should be detailed and clear.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response