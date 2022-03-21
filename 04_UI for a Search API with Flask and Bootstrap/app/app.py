from flask import Flask, Response, request, render_template
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util

# create an elasticsearch client
client = Elasticsearch("es:9200")

app = Flask(__name__)

@app.route('/search')
def search(index="ai", top_k=10, return_k=5):
	
	# get the query
	query = request.args.get('search', None)

	if query:
	
		# using a match query 
		body = {"size": return_k, "query": {"match": {"text": query}}}

		# submit the search query to elasticsearch
		docs = client.search(index=index, body=body)

		# reshape search results to prepare them for sentence embeddings
		texts = []
		section_titles = []
		article_titles = []
		score = []

		for d in docs["hits"]["hits"]:
			texts.append(d["_source"]["text"])
			section_titles.append(d["_source"]["section_title"])
			article_titles.append(d["_source"]["article_title"])
			score.append(d["_score"])

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
				'article_title': article_titles[idx],
				'section_title': section_titles[idx],
				'text': texts[idx],
				'bert_score': item['score']}
			rr_results_list.append(rr_results_dict)

		results = rr_results_list
	
	else:
		results = None

	return render_template("index.html", results=results)


if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True, port="5000")

# prevent cached responses (useful for debugging)
if app.config["DEBUG"]:
	@ app.after_request
	def after_request(response):
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
		response.headers["Expires"] = 0
		response.headers["Pragma"] = "no-cache"
		return response
	