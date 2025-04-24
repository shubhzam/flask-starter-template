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
from app.models import ConversationHistory
from app.extensions import db
from datetime import datetime
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings":True}
)
llm = OllamaLLM(model="gemma3:12b",
                n_predict=1024,
                temperature=0.1,
                keep_alive='-1' 
                )

PROMPT_TEMPLATE = """
You are an AI assistant that provides accurate answers **only** from the provided document context.
If the answer is not present, reply "I don't know." **(do not hallucinate).**

➤ **Write the answer directly** – no preambles like "Based on the context" and no meta commentary.
➤ **Be comprehensive**: paraphrase and include every relevant detail that appears in the context.
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

vectordb = None
def get_vectordb():
    """Get or initialize the vector database"""
    global vectordb
    if vectordb is None:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return vectordb

vectordb  = get_vectordb()                      
retriever = vectordb.as_retriever(               
    search_type="similarity",
    search_kwargs={"k": 5},
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = OllamaLLM(
        model="llama3.2-vision:latest",
        n_predict=1024,           
        temperature=0.1,
        keep_alive="-1s"          
    ),
    retriever = retriever,
    memory = memory,
    chain_type = "stuff",
    combine_docs_chain_kwargs = {"prompt": prompt},
    return_source_documents = True,
)

@chat_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started with chat"}), 200

@chat_bp.route('/', methods=['POST'])
def chat():
    data        = request.get_json(force=True)
    question    = data.get("question")
    doc_filter  = data.get("retrieved document")  
    convo_id = data.get("convo_id",1) 
    user_name = data.get("user_name", "shubham")

    if not question:
        return jsonify({"error": "`user query` is required"}), 400


    if doc_filter:
        retriever.search_kwargs["filter"] = {"source": doc_filter}
    else:
        retriever.search_kwargs.pop("filter", None)

    result = qa_chain({"question": question})

    if convo_id:  # If a conversation ID was provided
        conversation_record = ConversationHistory(
            convo_id=convo_id,
            human_message=question,
            ai_message=result["answer"],
            user_name=user_name,
            request_datetime=datetime.utcnow(),
            response_datetime=datetime.utcnow()
        )
        
        # Add and commit the record to the database
        db.session.add(conversation_record)
        db.session.commit()
    
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