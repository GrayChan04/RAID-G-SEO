import json
from pdb import set_trace as bp

beat = 0
result_dict = json.load(open('./result.json'))
improvements_sum = {}
for query in result_dict:
    i = result_dict[query]['idx'] # represent which one geo
    impressions_list = result_dict[query]['impressions_list']
    for impression in impressions_list: 
        improvements_sum_list = [ 0 for _ in range(11)] # 11 geo methods
        for idx, geo_method in enumerate(impression['improvements']): # idx represents geo method
            improvements_sum_list[idx] = improvements_sum_list[idx] + geo_method[i]
        improvements_sum[impression['impression_fn']] = {"sum_list": improvements_sum_list, "beat_list": [], "flag": 'True'}
    break
for impression in improvements_sum:
    for score in improvements_sum[impression]["sum_list"][:10]:
        if score > improvements_sum[impression]["sum_list"][-1]:
            improvements_sum[impression]["beat_list"].append(improvements_sum[impression]["sum_list"].index(score))
            improvements_sum[impression]["flag"] = 'False'
    print(impression, improvements_sum[impression])
    print("\n")
