from flask import Flask, Response
import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)


@ app.route('/<query>', methods=['GET'])
def search(query, index="pandemics", top_k=10, return_top_k=5):
    """
    1. Find best 50 results in ElasticSearch
    2. Use a transformers model to compute embeddings 
    3. Compute similarity scores between teh query embeddings and document embeddings
    4. Build a dictionary of the results, reranked
    """
    # create an elasticsearch client
    client = Elasticsearch("localhost:9200")

    # using a boolean query to exclude irrelevant sections
    exclude_sections = ["See also", 
                    "Further reading", 
                    "Data and graphs", 
                    "Medical journals", 
                    "External links"]
                    
    bool_query_body = {"size": return_top_k,
                        "query": {
                        "bool": {
                            "should": 
                            { "match": {"text": query}},
                            "must_not": {
                                "terms": {"section_title.keyword": exclude_sections}
                                },
                            }
                        }}

    # submit the search query to elasticsearch
    bool_docs = client.search(body = bool_query_body, index=index, size=5)

    # reshape search results to prepare them for sentence embeddings
    texts = []
    section_titles = []
    article_titles = []

    for doc in bool_docs["hits"]["hits"]:
        texts.append(doc["_source"]["text"])
        section_titles.append(doc["_source"]["section_title"])
        article_titles.append(doc["_source"]["article_title"])

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
            'text': texts[idx]
        }
        rr_results_list.append(rr_results_dict)
    
    results_dict = json.dumps({"query": query,
                               "results":  rr_results_list})

    return (Response(results_dict, status=422, mimetype='application/json'), 204)


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
