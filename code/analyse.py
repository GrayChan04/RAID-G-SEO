import json
from pdb import set_trace as bp


result_dict = json.load(open('./result_GLM-4-9B-0414_100_1+2+reflection+3+4_1226.json'))
improvements_sum = {}
impressions_list = [x for x in range (11)]
for impression in impressions_list:
    improvements_sum[impression] = [0 for _ in range (11)]

for id, index in enumerate(result_dict): # index: group, 5 query + avearge = 1 group
    i = result_dict[index]['idx'] # represent which one need optimization
    impressions_list = result_dict[index]['impressions']['average']['impression_list']
    for impression_num, impression in enumerate(impressions_list): 
        for idx, geo_method in enumerate(impression['improvements']): # idx: geo method
            improvements_sum[impression_num][idx] = improvements_sum[impression_num][idx] + geo_method[i]

for impression_scores in improvements_sum.values():
    print(impression_scores)


improvements_sum['work'] = [0 for _ in range (11)] # 11 geo methods
for id, group in enumerate(result_dict): # 5 query + avearge = 1 group
    # if id > 90:
    #     break
    i = result_dict[group]['idx'] # represent which one need optimization
    impressions = result_dict[group]['impressions']
    for query in impressions.keys():
        if query == 'average':
            break
        subjective_average_list = impressions[query]['impressions_list'][-1]['improvements']
        # subjective_average_list = impressions[query]['impressions_list'][2]['improvements']
        objective_average_list = impressions[query]['impressions_list'][2]['improvements']
        work_sub = False
        work_obj = False
        for method_num, method_scores in enumerate(subjective_average_list):
            if method_scores[i] < 0:
                # improvements_sum['work'][method_num] += 1
                work_sub = True
        for method_num, method_scores in enumerate(objective_average_list):
            if method_scores[i] < 0:
                # improvements_sum['work'][method_num] += 1
                work_obj = True
        if work_sub and work_obj:
            print(query)
            
# print(improvements_sum)