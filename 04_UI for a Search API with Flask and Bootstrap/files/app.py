from flask import Flask, jsonify, make_response, Response, render_template, request
import json
import requests
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
@app.route("/")
def home():
	return render_template("index.html")

@app.route('/<query>', methods=['GET'])
def search(query, index="ai", top_k=10, return_k=50):
	
	# create an elasticsearch client
	client = Elasticsearch()
	
	# using a match query 
	body = {"size": return_k, "query": {"match": {"text": query}}}

	# submit the search query to elasticsearch
	results = client.search(index=index, body=body)

	# reshape search results to prepare them for sentence embeddings
	texts = []
	section_titles = []
	article_titles = []
	score = []

	for r in results["hits"]["hits"]:
		texts.append(r["_source"]["text"])
		section_titles.append(r["_source"]["section_title"])
		article_titles.append(r["_source"]["article_title"])
		score.append(r["_score"])

	# using the distilled BERT model
	model='distilbert-base-nli-stsb-mean-tokens'
	emb = SentenceTransformer(model)
	corpus_emb = emb.encode(texts, convert_to_tensor=True)
	query_emb = emb.encode(query, convert_to_tensor=True)
	
	# returns a list of dictionaries with the keys 'corpus_id' and 'score',
	# sorted by decreasing cosine similarity scores
	reranked_results = util.semantic_search(query_emb, corpus_emb, top_k=top_k)[0]

	rr_results_list = []
	for item in reranked_results:
		idx = item['corpus_id']
		rr_results_dict = {
		'bert_score': item['score'],
			'article_title': article_titles[idx],
			'section_title': section_titles[idx],
			'text': texts[idx]}
		rr_results_list.append(rr_results_dict)

	results_dict = json.dumps({"query": query,
							"results":  rr_results_list})

	return Response(results_dict, status=422, mimetype='application/json')


if __name__ == '__main__':
	app.run(debug=True)

# prevent cached responses (useful for debugging)
if app.config["DEBUG"]:
	@ app.after_request
	def after_request(response):
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
		response.headers["Expires"] = 0
		response.headers["Pragma"] = "no-cache"
		return response
	