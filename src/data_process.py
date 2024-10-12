from datasets import load_dataset
import json

if __name__ == '__main__':
    dataset = load_dataset("GEO-Optim/geo-bench") # dataset size = 1000
    json_total = {}
    query_repeat = 0
    sources_repeat = 0
    for index, data in enumerate(dataset['train']):
        sources = []
        responses = []
        for source in data['sources']:
            source_item = {"url": source['url'], "raw_source": source["raw_text"], "source": source["cleaned_text"], "summary": source["cleaned_text"][:8000]}
            sources.append(source_item)
        json_total[data['query']] = [{"sources": sources, "responses": responses}]
    json.dump(json_total, open('./global_cache_original.json', 'w'), indent = 2) # cache size = 989, repeat_num = 11