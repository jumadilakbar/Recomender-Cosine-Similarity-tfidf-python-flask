from flask import Flask, request, jsonify
import pandas as pd
import pickle
# from cosine import Cosine
from tfidf import Engine
app = Flask(__name__)

#author: Muhamad Jumadil Akbar
#Email : muhamadjumadilakbar@gmail.com
#fullstak Developer 

@app.route('/')
def index():
    return jsonify(message='success',app='Cosine Similarity')

@app.route('/api')
def api():
    # file_pi2 = open('Engine.obj', 'r')
    # engine = pickle.load(file_pi2)
    engine = Engine()
    target_title=str(request.args.get('title'))
    # target_tags=str(request.args.get('tags'))

    databuku = pd.read_csv('test.csv')

    match_tags = []
    match_tags_full = []
    match_title = []
    data = {}
    data['title'] = databuku.preprocessed.tolist()
    data['outtitle'] = databuku.Judul.tolist()
    
    engine.setQuery(target_title)
    for title in data['title']:
        engine.addDocument(title)
    #
    #targets = data['title'][int(title)]
    titles_score = engine.process_score()

    

    return jsonify(skor=titles_score)
    # return jsonify(match=match_tags, tag=target_tags, tags=num)

app.run(debug=True)
