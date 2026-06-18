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
from langchain_community.retrievers import BM25Retriever,EnsembleRetriever
import os




load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=ChatGroq(model="llama-3.3-70b-versatile")



embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

if os.path.exists("faiss_index"):
    print("loading existing faiss index....")
    vec_store=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
else:
    docs=PyMuPDFLoader("document\\toaz.info-india-national-building-code-nbc-2016-vol-1pdf-pr_26cbd1ad8b437409f28176f6fefcbb10.pdf").load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitter=text_splitter.split_documents(docs)
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

response=chat.invoke({"input":"What are the requirements for fire safety in residential buildings according to the NBC 2016?"})
print(response['answer'])

"""eval_data=[{
    "question":"What is the minimum staircase width?",
    "ground_truth":"The minimum staircase width is 1 meter."
},
{
    "question":"What are the fire exit requirements?",
    "ground_truth":"Fire exits must comply with NBC Part 4."
}]


# ==========================================
# Evaluation Dataset
# ==========================================

test_cases = [
    {
        "question": "What are the fire safety requirements for residential buildings?",
        "expected_keywords": ["fire", "safety", "exit"]
    },
    {
        "question": "What are the requirements for emergency exits?",
        "expected_keywords": ["exit", "emergency"]
    },
    {
        "question": "What are the requirements for ventilation in buildings?",
        "expected_keywords": ["ventilation", "air"]
    },
    {
        "question": "What are the accessibility requirements for public buildings?",
        "expected_keywords": ["accessibility", "disabled"]
    },
    {
        "question": "What are the requirements for water supply systems?",
        "expected_keywords": ["water", "supply"]
    }
]


# ==========================================
# Retrieval Evaluation
# ==========================================

total_questions = len(test_cases)

correct_retrievals = 0
precision_scores = []

for test in test_cases:

    question = test["question"]
    expected_keywords = test["expected_keywords"]

    retrieved_docs = retriever.invoke(question)

    relevant_chunks = 0

    found_relevant = False

    for doc in retrieved_docs:

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in expected_keywords
        ):
            relevant_chunks += 1
            found_relevant = True

    if found_relevant:
        correct_retrievals += 1

    precision = relevant_chunks / len(retrieved_docs)

    precision_scores.append(precision)

    print("=" * 80)
    print("QUESTION:")
    print(question)

    print("\nPRECISION:")
    print(round(precision, 2))

    print("\nTOP RETRIEVED CHUNKS:\n")

    for idx, doc in enumerate(retrieved_docs[:2]):
        print(f"Chunk {idx+1}")
        print(doc.page_content[:400])
        print("\n")


# ==========================================
# Retrieval Metrics
# ==========================================

retrieval_recall = (
    correct_retrievals / total_questions
)

avg_precision = (
    sum(precision_scores)
    / len(precision_scores)
)

print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)

print(
    f"Recall@K: {retrieval_recall:.2f}"
)

print(
    f"Average Precision@K: {avg_precision:.2f}"
)


# ==========================================
# Hallucination Evaluation
# ==========================================

out_of_scope_questions = [
    "Who is the Prime Minister of India?",
    "What is the salary of a civil engineer?",
    "Who won IPL 2025?",
    "What is the capital of France?",
    "Who founded Microsoft?"
]

hallucinations = 0

for question in out_of_scope_questions:

    response = chat.invoke(
        {"input": question}
    )

    answer = response["answer"].lower()

    print("\n" + "=" * 80)
    print("OUT OF SCOPE QUESTION:")
    print(question)

    print("\nANSWER:")
    print(response["answer"])

    if (
        "could not find" not in answer
        and "not available" not in answer
        and "provided context" not in answer
        and "building code documents" not in answer
    ):
        hallucinations += 1


hallucination_rate = (
    hallucinations
    / len(out_of_scope_questions)
)

print("\n" + "=" * 80)
print("HALLUCINATION EVALUATION")
print("=" * 80)

print(
    f"Hallucination Rate: {hallucination_rate:.2f}"
)

# ==========================================
# Final Score
# ==========================================

overall_score = (
    (
        retrieval_recall
        + avg_precision
        + (1 - hallucination_rate)
    )
    / 3
)

print("\n" + "=" * 80)
print("FINAL SCORE")
print("=" * 80)

print(
    f"Overall RAG Score: {overall_score:.2f}"
)"""