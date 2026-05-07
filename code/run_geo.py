from utils import get_answer, extract_citations_new, impression_subjective_impression, impression_wordpos_count_simple, impression_subjpos_detailed, impression_diversity_detailed, impression_uniqueness_detailed, impression_follow_detailed, impression_influence_detailed, impression_relevance_detailed, impression_subjcount_detailed, impression_pos_count_simple, impression_word_count_simple
from typing import List, Tuple
import numpy as np
import json
# from src.geo_functions import *
from geo_functions import *
import sys
import time
import os
from datasets import load_dataset, load_from_disk, Dataset, Sequence, Value
from pdb import set_trace as bp
import time

# from openai import OpenAI
# client = OpenAI(api_key = 'sk-7hyJmJEhn2MYBPlR16C6163b4a3546A8A021D1413c6416B3', base_url = 'https://api.yxflow.xyz/v1')
# client = OpenAI(api_key = 'sk-LH2LJOrZh1kZx9A0PQAw89JZiqzJ9b6UypYkyIKNpWXPwQjL', base_url = 'https://dzqc.link/v1/')

# from zhipuai import ZhipuAI
# client = ZhipuAI(api_key="880f73cd7d98f9981b1b8a80dd1136c6.L4mk4JObnqYJ7USZ")

def identity(summary):
	return summary

IMPRESSION_FNS = {
	'simple_word' : impression_word_count_simple,
	'simple_pos' : impression_pos_count_simple,
	'simple_wordpos' : impression_wordpos_count_simple, 
	'relevance_detailed' : impression_relevance_detailed,
	'influence_detailed' : impression_influence_detailed,
	'uniqueness_detailed' : impression_uniqueness_detailed,
	'diversity_detailed' : impression_diversity_detailed,
	'follow_detailed' : impression_follow_detailed,
	'subjpos_detailed' : impression_subjpos_detailed,
	'subjcount_detailed' : impression_subjcount_detailed,
	'subjective_score' : impression_subjective_impression,
}


GEO_METHODS = {
	'identity' : identity,
	'seo_optimize_mine2' : seo_optimize_mine2, # Keyword Stuffing
	'unique_words_gpt' : unique_words_optimization_gpt,
	'simple_language_mine': simple_language_mine,
	'authoritative_mine' : authoritative_optimization_mine,
	'fluent_gpt' : fluent_optimization_gpt,
	'technical_terms_mine' : technical_terms_mine,
	'citing_credible_mine': citing_credible_sources_mine,
	'more_quotes_mine' : more_quotes_mine,
	'stats_optimization_gpt' : stats_optimization_mine,
	# 'intent_optimization_1234' : intent_optimization_1234,
	'intent_optimization_1_234' : intent_optimization_1_234,
	# 'intent_optimization_12_34' : intent_optimization_12_34,
	# 'intent_optimization_123_4' : intent_optimization_123_4,
	# 'intent_optimization_1_2_34' : intent_optimization_1_2_34,
	# 'intent_optimization_12_3_4' : intent_optimization_12_3_4,
	# 'intent_optimization_1_23_4' : intent_optimization_1_23_4,
	# 'intent_optimization_1_2_3_4' : intent_optimization_1_2_3_4,
	'intent_optimization_234' : intent_optimization_234,
	# 'intent_optimization_2_34' : intent_optimization_2_34,
	'intent_optimization_34' : intent_optimization_34,
	# 'intent_optimization_3_4' : intent_optimization_3_4,
	'intent_optimization_4' : intent_optimization_4,
	# 'intent_optimization_1_2rq34': intent_optimization_1_2rq34,
	# 'intent_optimization_1_2rq_34': intent_optimization_1_2rq_34,
	# 'intent_optimization_1_2rq_234': intent_optimization_1_2rq_234,
	# 'intent_optimization_1_2rq_234_3': intent_optimization_1_2rq_234_3,
	# 'intent_optimization_1_2rq_234_rq': intent_optimization_1_2rq_234_rq,
	# 'intent_optimization_1_2rq_234_3rq': intent_optimization_1_2rq_234_3rq,
	# 'intent_optimization_1_2rq_234_new': intent_optimization_1_2rq_234_new,
	# 'intent_optimization_1_2rq_234_3rqinsight': intent_optimization_1_2rq_234_3rqinsight,
	# 'intent_optimization_1_2rq_234_3rqinsightnew': intent_optimization_1_2rq_234_3rqinsightnew,
	# 'intent_optimization_1_2rq_234_3insight': intent_optimization_1_2rq_234_3insight,
	# 'intent_optimization_1_2reflection_234': intent_optimization_1_2reflection_234,
	# 'intent_optimization_1_2reflection34':intent_optimization_1_2reflection34,
	'intent_optimization_1_2reflection_234_3': intent_optimization_1_2reflection_234_3,
	# 'intent_optimization_2reflection_234_3': intent_optimization_2reflection_234_3,
	# 'intent_optimization_1_34': intent_optimization_1_34,
	# 'intent_optimization_1_2reflection_24_4': intent_optimization_1_2reflection_24_4,
	# 'intent_optimization_1_2reflection_234_3_rolestep': intent_optimization_1_2reflection_234_3_rolestep
}

EXTRACTIVE = False
loaded_cache = None
LAST_UPDATE_TIME = time.time()

def improve(query : str, idx : int, sources : List[str] = None, summaries : List[str] = None, impression_fn = impression_wordpos_count_simple, returnFullData = False, static_cache=os.environ.get('STATIC_CACHE', None)=='True') -> Tuple[np.array, List]: 
	global loaded_cache
	global LAST_UPDATE_TIME
	if static_cache:
		if loaded_cache is not None:
			modified_time = os.path.getmtime(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json'))
			if modified_time - LAST_UPDATE_TIME > 0:
				loaded_cache = json.load(open(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')))
			LAST_UPDATE_TIME = 	modified_time

			pass
		else:
			loaded_cache = json.load(open(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')))
	else:
		loaded_cache = None
	# idx indicates the website to boost
	# print('query is', query)
	answers = get_answer(query, summaries = summaries, num_completions = 5, n = 5, loaded_cache = loaded_cache)
	if sources is None:
		sources = [x['source'] for x in answers['sources']]
	if summaries is None:
		summaries = [x['summary'] for x in answers['sources']]
	answers = answers['responses'][-1]
	# print("original answers:", answers)
	if impression_fn == impression_subjective_impression or  impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
		orig_init_scores = np.array([impression_fn(x, query, 5, idx = idx) for x in answers])
		orig_init_scores = orig_init_scores[~np.all(orig_init_scores == 0, axis=1)]
	else:
		orig_init_scores = np.array([impression_fn(extract_citations_new(x), 5) for x in answers])
	init_scores = orig_init_scores.mean(axis=0)
	print('Init Scores: ', init_scores)
	improvements = []
	all_final_scores = []

	for meth_name in GEO_METHODS:
		# print(meth_name + ":")
		summaries_copy = summaries[:idx] + [GEO_METHODS[meth_name](summaries[idx])] + summaries[idx+1:]
		answers = get_answer(query, summaries = summaries_copy, num_completions = 5, n = 5, loaded_cache = loaded_cache)
		answers = answers['responses'][-1]
		# print("optimizated answers:", answers)
		if impression_fn == impression_subjective_impression or impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
			final_scores = np.array([impression_fn(x, query, 5, idx = idx) for x in answers])
			final_scores = final_scores[~np.all(final_scores == 0, axis=1)]
		else:
			final_scores = [impression_fn(extract_citations_new(x), 5) for x in answers]
		all_final_scores.append(np.array(final_scores))
		final_scores = np.array(final_scores).mean(axis=0)
		print(meth_name + ":", final_scores)
		# print('Optimization Score: ', final_scores)
		improvements.append(((final_scores - init_scores) / (init_scores + 1)))
	improvements = np.vstack(improvements)

	if returnFullData:
		# return orig_init_scores, all_final_scores
		return orig_init_scores, all_final_scores
	else:
		return improvements, improvements[:, idx] > 0


# def modify_dataset(k):
# 	# k['query'] = str
# 	prompt = """You are a user of the Generative Search Engine, seeking to explore a specific topic. You will be given a query representing what you want to learn about. Your task is to generate 4 new queries that approach the topic from different perspectives. Ensure that: 
# 1. Each query is relevant to the original but not a direct repetition.
# 2. The queries cover a range of perspectives, such as related concepts, practical applications, contrasting viewpoints, emerging trends, or any other meaningful dimension. But do not limit to these directions.
# 3. Output ONLY 1 query per line.

# ### Input Format
# {query_original}

# ### Output Format
# query_1
# query_2
# query_3
# query_4""".format(query_original = k['query'])
# 	response = client.chat.completions.create(
# 		model = 'gpt-4',
# 		# model = "glm-4-flash", 
# 		temperature = 0.5,
# 		max_tokens = 1024,
# 		messages = [
# 			# { 'role': "system", 'content': system_prompt },
# 			{ 'role': "user", 'content': prompt }
# 		],
# 		top_p = 1,
# 		n = 1,
# 	)
# 	# queries = response['choices'][0]['message']['content']
# 	queries = response.choices[0].message.content
# 	queries = queries.replace('\n\n','\n')
# 	queries = queries.split('\n')
# 	query_list = []
# 	query_list.append(k['query'])
# 	for query in queries:
# 		if 'query' not in query:
# 			query_list.append(query)
# 	k['query'] = query_list
# 	return k

if __name__ == '__main__':
	# # modify dataset
	# dataset = load_dataset(
	# 	"GEO-Optim/geo-bench", 
	# 	'test')
	# dataset = dataset['test']
	# dataset.save_to_disk("./datasets/dataset_original")
	# dataset = load_from_disk("./datasets/dataset_original")
	# modified_dataset = dataset.map(modify_dataset, batched = False, load_from_cache_file=False)
	# modified_dataset = modified_dataset.cast_column('query', feature = Sequence(Value("string")))
	# modified_dataset.save_to_disk("./datasets/dataset_queryAdding")
	dataset = load_from_disk("./datasets/dataset_queryAdding")
	# shuffle and split 100 for try
	split = dataset.train_test_split(
		test_size = 0.1,
		shuffle = True,
		seed = 42,
	)
	dataset = split['test']
	# print(len(dataset[0]['query'])) # exzamine

	result_dic = {}
	for i, k in enumerate(dataset):
		if i == 76 or i == 92:
			continue
		# Insert Metric here 
		idx = k['sugg_idx']
		result_dic[i] = {'idx': idx}
		query_impression = {}
		for query in k['query']:
			print(i, idx, query)
			query_impression[query] = {'impressions_list':[]}
			for impression in IMPRESSION_FNS:
				try:
					print(impression+ ":")
					improvements, progress = improve(query = query, idx = idx, impression_fn = IMPRESSION_FNS[impression]) # all methods are caculated 
					query_impression[query]['impressions_list'].append({'impression_fn': impression, 'improvements': improvements.tolist(), 'progress': progress.tolist()})	
				except Exception as e:
					print('Error',e)
					print(i, idx, query)
					continue
			len_impressions_list = len(query_impression[query]['impressions_list']) # num of impression = 11
			len_methods_list = len(query_impression[query]['impressions_list'][0]['improvements']) # num of methods = 11
			print(len_impressions_list, len_methods_list)
		average_dic_list = []
		for num_impression in range(len_impressions_list): # which impression
			improvement_average_list = []
			for num_method in range(len_methods_list): # which method
				improvements_list = [x['impressions_list'][num_impression]['improvements'][num_method] for x in query_impression.values()]
				impression_name = set(x['impressions_list'][num_impression]['impression_fn'] for x in query_impression.values()).pop()
				improvements_average = [sum(improvements) / len(improvements) for improvements in zip(*improvements_list)]
				improvement_average_list.append(improvements_average)
			average_dic_list.append({'impression_fn': impression_name, 'improvements': improvement_average_list})
		query_impression['average'] = {'impression_list': average_dic_list}
		result_dic[i]['impressions'] = query_impression
	json.dump(result_dic, open('./result_GLM-4-9B-0414_100_allmethods_260317.json', 'w'), indent = 2)
		