import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cuda"}
)
llm = OllamaLLM(model="llama3.2-vision:latest",
                n_predict=1024,
                temperature=0.1 
                )

PROMPT_TEMPLATE = """
You are an AI assistant that provides accurate answers **only** from the provided document context.
If the answer is not present, reply "I don't know." **(do not hallucinate).**

➤ **Write the answer directly** – no preambles like "Based on the context" and no meta commentary.
➤ **Be comprehensive**: include every relevant detail that appears in the context.
➤ Do **not** add information that is not explicitly in the context.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started with chat"}), 200

@chat_bp.route('/', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    question = data.get("question")
    retrieved_document = data.get("retrieved document")
    if not question:
        return jsonify({"error": "`user query` is required"}), 400
    
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    search_kwargs = {"k": 8}
    if retrieved_document:
        search_kwargs["filter"] = {"source": retrieved_document}

    base_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,  # keep the original query as well
        # generate 4 paraphrased queries
    )    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'  
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        memory=memory,
        chain_type="stuff",               
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    
    result = qa({"question": question})
    
    chat_history = []
    if "chat_history" in result:
        for msg in result["chat_history"]:
            if hasattr(msg, 'content'):
                role = "human" if hasattr(msg, 'example') and msg.example == False else "assistant"
                chat_history.append({"role": role, "content": msg.content})
    
    return jsonify({
        "answer": result.get("answer", ""),  
        "sources": [
            {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", ""),
                "chunk": doc.metadata.get("chunk", "")
            }
            for doc in result.get("source_documents", [])
        ],
        "chat_history": chat_history  
    }), 200