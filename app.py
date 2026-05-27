from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

docs=PyPDFLoader("document\\toaz.info-india-national-building-code-nbc-2016-vol-1pdf-pr_26cbd1ad8b437409f28176f6fefcbb10.pdf").load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter=text_splitter.split_documents(docs)
embeddings=OllamaEmbeddings(model='nomic-embed-text-v2-moe')
