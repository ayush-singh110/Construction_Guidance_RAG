from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import os




load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=ChatGroq(model="llama-3.3-70b-versatile")



embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

docs=PyMuPDFLoader("document\\toaz.info-india-national-building-code-nbc-2016-vol-1pdf-pr_26cbd1ad8b437409f28176f6fefcbb10.pdf").load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter=text_splitter.split_documents(docs)

if os.path.exists("faiss_index"):
    print("loading existing faiss index....")
    vec_store=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
else:
    vec_store=FAISS.from_documents(splitter,embeddings)
    vec_store.save_local("faiss_index")

retriever=vec_store.as_retriever(search_kwargs={"k":5})
bm_retriever=BM25Retriever.from_documents(splitter)
hybrid_retriever=EnsembleRetriever(retrievers=[retriever, bm_retriever], weights=[0.6, 0.4])
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
chat=create_retrieval_chain(hybrid_retriever,chain_doc)


app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json() or {}
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = chat.invoke({"input": question})
        answer = response.get('answer', '')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8501))
    app.run(host='0.0.0.0', port=port, debug=True)