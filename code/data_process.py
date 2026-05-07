from datasets import load_from_disk
import json


# if __name__ == '__main__':
#     dataset = load_from_disk("./datasets/dataset_queryAdding") # dataset size = 1000, num_query = 5000
#     json_total = {}
#     query_repeat = 0
#     sources_repeat = 0
#     for index, data in enumerate(dataset):
#         for query in data['query']:
#             sources = []
#             responses = []
#             for source in data['sources']:
#                 source_item = {"url": source['url'], "raw_source": source["raw_text"], "source": source["cleaned_text"], "summary": source["cleaned_text"][:8000]}
#                 sources.append(source_item)
#             json_total[query] = [{"sources": sources, "responses": responses}]
#     json.dump(json_total, open('./global_cache_original.json', 'w'), indent = 2) # cache size = 989, repeat_num = 11


# import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    dataset = load_from_disk("./datasets/dataset_queryAdding") # dataset size = 1000, num_query = 5000
    query_similarity = 0
    # for data in random.choices(dataset, k=5):
    #     print(data['query'])
    for data in dataset:
        queries = data['query']
        # for query in queries:
            # query_len += len(query.split())
        for query in queries[1:]:
            corpus = [queries[0], query]
            vectorizer = CountVectorizer().fit_transform(corpus)
            similarity = cosine_similarity(vectorizer)[0][1]
            print(cosine_similarity(vectorizer))
            query_similarity += similarity
    print(query_similarity / 4000)