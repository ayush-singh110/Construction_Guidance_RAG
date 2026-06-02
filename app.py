from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=ChatGroq(model="llama-3.3-70b-versatile")
docs=PyMuPDFLoader("document\\toaz.info-india-national-building-code-nbc-2016-vol-1pdf-pr_26cbd1ad8b437409f28176f6fefcbb10.pdf").load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter=text_splitter.split_documents(docs)
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vec_store=FAISS.from_documents(splitter,embeddings)
retriever=vec_store.as_retriever()
template="""
You are a professional Building Code Advisor AI.

Answer the user's question strictly using the provided context.

Guidelines:
- Cite section numbers when available
- Mention source document/page if present
- Be concise but accurate
- If answer is unavailable, clearly state that
- Never generate fake building regulations

Context:
{context}

Question:
{input}
"""

prompt=ChatPromptTemplate.from_messages([("system", template),("human", "{input}")])

chain_doc=create_stuff_documents_chain(llm,prompt)
chat=create_retrieval_chain(retriever,chain_doc)

response=chat.invoke({"input":"What are the requirements for fire safety in residential buildings according to the NBC 2016?"})
print(response['answer'])

