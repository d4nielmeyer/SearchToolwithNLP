
import json
import wikipediaapi
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch, helpers
from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import wikipediaapi

client = Elasticsearch({"scheme": "http", "host": "localhost", "port": 9200})

# initialize Wikipedia object
wiki_data = wikipediaapi.Wikipedia('en')

# to get all pages from a given category (here: artificial intelligence), use property categorymembers
# returns all members of given category


def create_index(index_name: str="ai"):
    client.indices.create(index=index_name)

def populate_index(
    index_name: str="ai",
    data_path: str="data.json"
    ):

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    for d in tqdm(data, position=0, leave=True):
        client.index(index=index_name, body=d)


if __name__=="__main__":
    populate_index()