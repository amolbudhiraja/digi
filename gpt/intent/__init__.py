import yaml
import asyncio
import aiofiles
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify, make_response
from sentence_transformers import SentenceTransformer
from QueryProcessing import QueryProcessor

app: Flask = Flask(__name__)
CORS(app)

app = Flask(__name__)

@app.route('/query', methods=['GET'])
def handle_query() -> None:
    """
    Handles the user's search query.

    Returns:
        A list of 3 digis which relate to the user's search query the most. (Relation is based on cosine similarity.)
    """
    user_query: str = request.args.get('search', default='')
    DATA_FOLDER: str = "../catalog/"
    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    homes: list = loop.run_until_complete(QueryProcessor.get_homes(DATA_FOLDER, 5))
    concatenated_data: list = []
    for home in homes:
        home_name: str = home['metadata']['name']
        objects_names: list = QueryProcessor.extract_names_from_mount(home['spec']['mount'])
        concatenated_string: str = home_name + ' ' + ' '.join(objects_names)
        concatenated_data.append(concatenated_string)
    user_vector: np.ndarray = QueryProcessor.vectorize(user_query)
    digi_vectors: np.ndarray = QueryProcessor.vectorize(concatenated_data)
    top_matches: list = QueryProcessor.similarity_search(user_vector, digi_vectors, 3)
    matched_homes: list = [homes[i] for i in top_matches]
    response: make_response = make_response(jsonify({"homes": matched_homes}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/', methods=['GET'])
def home_page() -> str:
    """Default web service route.
    
    Returns:
        A welcome message and instructures on accessing other endpoints in the service. 
    """
    return "Welcome to Digi Intent Processing Web Service. Use /query?search=<search query> to receive the digis assocaited with the particular search query."

if __name__ == "__main__":
    app.run(debug=True)
