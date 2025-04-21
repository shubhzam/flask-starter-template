from flask import Blueprint, request, jsonify
#from app.extensions import db

rag_bp = Blueprint('rag', __name__)

rag_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message":"getting started"}), 200

@rag_bp.route('/', methods=['POST'])
def create_item():
    payload = request.get_json(force=True)
    user_input = payload.get('user_input')
    document_b64 = payload.get('document_b64') 
    if not user_input and document_b64:
        return jsonify(error='`user_input` and `document_b64` is required'), 400

    return jsonify({"message":"working"}), 201
