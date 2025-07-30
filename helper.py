import os
from dotenv import load_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
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

