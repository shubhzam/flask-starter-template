import os
import base64
from typing import List
from flask import Blueprint, request, jsonify
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./_chromadb")
#from app.extensions import db
device = "cuda" if torch.cuda.is_available() else "cpu"
vectordb_bp = Blueprint('vectordb', __name__)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2",model_kwargs={"device": "cuda"})

@vectordb_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message":"getting started"}), 200

@vectordb_bp.route('/', methods=['POST'])
def rag():
        # payload = request.get_json(force=True)
        # user_input = payload.get('user_input') 
        # if not user_input:
            # return jsonify(error='`user_input` and `document_b64` is required'), 400
    pdf_folder = "data"
    pdf_files = [
        os.path.join(pdf_folder, fn)
        for fn in os.listdir(pdf_folder)
        if fn.lower().endswith(".pdf")
    ]
    raw_docs: List[Document] = []
    basename_list: List = []  
    for path in pdf_files:
        basename = os.path.basename(path)
        loader = UnstructuredPDFLoader(path)
        pages  = loader.load()
        for d in pages:
            d.metadata["source"] = basename
            basename_list.append(basename)
            raw_docs.append(d)
    # serializable_docs = [
    #     {
    #         "text": d.page_content,       # or slice d.page_content[:200] if it’s huge
    #         "metadata": d.metadata
    #     }
    #     for d in raw_docs
    # ]

    # custom split function by abhiraj
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    chunked_docs: List[Document] = []
    for doc in raw_docs:
        texts = splitter.split_text(doc.page_content)
        for idx, txt in enumerate(texts):
            chunked_docs.append(
                Document(
                    page_content=txt,
                    metadata={
                        "source": doc.metadata["source"],
                        "chunk":  idx
                    }
                )
            )
    print(chunked_docs[0], chunked_docs[1], chunked_docs[3])

    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    

    return jsonify({"success":"vector db created",
                    "retrieved document":basename_list}), 201
