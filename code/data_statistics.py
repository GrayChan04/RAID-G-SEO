import json
from collections import Counter
from pdb import set_trace as bp

# statistics for success data
purpose_counter = Counter()
action_counter = Counter()
role_counter = Counter()

with open('steps_classification_260106.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        if 'Purpose' in entry and 'Action' in entry:
            entry_new = {entry['Purpose']: entry['Action']}
        else:
            entry_new = entry
        for purpose, action in entry_new.items():
            purpose_counter[purpose] += 1
            action_counter[action] += 1

with open('role_260106.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        for role, _ in entry.items():
            role_counter[role] += 1
            break # ignore reflection
print(role_counter)
print(len(role_counter.keys()))
print(sum(role_counter.values()))
print(purpose_counter)
print(action_counter)
print(sum(purpose_counter.values()))
print(sum(action_counter.values()))
# print(role_counter)
# plot
# fig, axs = plt.subplots(1, 2, figsize=(14, 7))
# # left
# axs[0].pie(
#     list(purpose_counter.values()),
#     labels=list(purpose_counter.keys()),
#     autopct='%1.1f%%',
#     startangle=140
# )
# axs[0].set_title('Distribution of Purposes')
# axs[0].axis('equal')
# # right
# axs[1].pie(
#     list(action_counter.values()),
#     labels=list(action_counter.keys()),
#     autopct='%1.1f%%',
#     startangle=140
# )
# axs[1].set_title('Distribution of Actions')
# axs[1].axis('equal')
# plt.tight_layout()
# plt.savefig("pie_chart_distribution_puerpose_action.png", dpi=300)
# # # left
# axs[0].pie(
#     list(role_counter.values()),
#     labels=list(role_counter.keys()),
#     autopct='%1.1f%%',
#     startangle=140
# )
# axs[0].set_title('Distribution of Purposes')
# axs[0].axis('equal')
# plt.tight_layout()
# plt.savefig("pie_chart_distribution_role.png", dpi=300)





# from datasets import load_from_disk

# dataset = load_from_disk("./datasets/dataset_queryAdding")
# # shuffle and split 100 for try
# split = dataset.train_test_split(
#     test_size = 0.1,
#     shuffle = True,
#     seed = 42,
# )
# dataset = split['test']
# for i, query in enumerate(dataset[40]['query']):
#     print(i)
#     print(query)

# for k, source in enumerate(dataset[40]['sources']):
#     print(k)
#     print(source['cleaned_text'])