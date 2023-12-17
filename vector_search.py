import pinecone
from sentence_transformers import SentenceTransformer,util
import os

pineode_api_key=os.environ["PINECONE_API_KEY"]
pinecode_index=os.environ["PINECONE_INDEX"]
pinecode_index=os.environ["PINECONE_ENVIRONMENT"]



model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=pineode_api_key, environment=pinecode_index) 
index = pinecone.Index(pinecode_index)

def addData(corpusData,url):
    id = id = index.describe_index_stats()['total_vector_count']
    for i in range(len(corpusData)):
        chunk=corpusData[i]
        chunkInfo=(str(id+i),
                model.encode(chunk).tolist(),
                {'title': url,'context': chunk})
        index.upsert(vectors=[chunkInfo])

def find_match(query,k):
    query_em = model.encode(query).tolist()
    result = index.query(query_em, top_k=k, includeMetadata=True)
    
    return [result['matches'][i]['metadata']['title'] for i in range(k)],[result['matches'][i]['metadata']['context'] for i in range(k)]