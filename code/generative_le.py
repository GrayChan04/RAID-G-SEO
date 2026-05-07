import openai
import time
import os
import pickle
import uuid
from pdb import set_trace as bp

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

# llama-3-8B-Instruct & GLM-4-9B-0414
# import torch

from model_load import model_get

# model set
model_name = "/home/xchen/models/GLM-4-9B-0414" # Meta-Llama-3-8B-Instruct, GLM-4-9B-0414

model_, tokenizer, terminators = model_get(model_name = model_name)


def generate_answer(query, sources, num_completions, temperature = 0.5, verbose = False, model = 'gpt-3.5-turbo-16k'): 

    # openai.api_base = 'https://api.openai.com/v1'
    
    source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query = query, source_text = source_text)

    model = model_

    while True:
        try:
            # print('Running OpenAI Model')
            # # print('Running zhipu Model')
            # # response = openai.ChatCompletion.create( # desert
            # response = client.chat.completions.create(
            #     model = model,
            #     # model = "glm-4-flash", 
            #     temperature=temperature,
            #     max_tokens=1024,
            #     messages = [
            #         # { 'role': "system", 'content': system_prompt },
            #         { 'role': "user", 'content': prompt }
            #     ],
            #     top_p=1,
            #     n=num_completions,
            # )
            print('Running GLM-4-9B-0414 Model') # Llama-3-8B-Instruct, GLM-4-9B-0414
            input_ids = tokenizer.apply_chat_template(
                [
                    # { 'role': "system", 'content': system_prompt },
                    { 'role': "user", 'content': prompt }
                ],
                add_generation_prompt = True,
                return_tensors = "pt"
            ).to(model.device)
            response = model.generate(
                input_ids,
                max_new_tokens = 1024,
                eos_token_id = terminators,
                do_sample = True,
                temperature = temperature,
                top_p = 1,
                num_return_sequences = num_completions
            )
            print('Response Done')
            break
        except Exception as e:
            # print('Error in calling OpenAI API', e)
            # print('Error in calling zhipu API', e)
            # print('Error in calling Llama API', e)
            print('Error in calling GLM API', e)
            time.sleep(15)
            continue
    # pickle.dump(response.usage, open(f"response_usages_gpt16k/{uuid.uuid4()}.pkl", "wb"))
    # pickle.dump(response.usage, open(f"response_usages_zhipu/{uuid.uuid4()}.pkl", "wb"))
    # pickle.dump(response, open(f"response_usages_Llama-3-8B-Instruct/{uuid.uuid4()}.pkl", "wb"))
    pickle.dump(response, open(f"response_usages_GLM-4-9B-0414/{uuid.uuid4()}.pkl", "wb"))
    # return [x.message.content + '\n' for x in response.choices]
    return [tokenizer.decode(x[input_ids.shape[-1]:], skip_special_tokens = True) + '\n' for x in response]    