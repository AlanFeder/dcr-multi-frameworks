import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pyprojroot import here

load_dotenv()

with open(
    "/Users/alanfeder/Documents/talks/dcr-3-frameworks/data/external/transcripts.pkl", "rb"
) as f1:
    tx1 = pickle.load(f1)



df_info = pd.read_parquet(
    "/Users/alanfeder/Documents/talks/dcr-3-frameworks/data/external/talks_on_youtube.parquet"
)

dcr_info = df_info[df_info["id0"].str.slice(0, 3) == "DCR"]
dcr_info = dcr_info[~(dcr_info["Speaker"].isna())]
dcr_info = dcr_info[
    ["id0", "Year", "Speaker", "Title", "VideoURL", "Abstract"]
].reset_index(drop=True)

dcr_info2 = dcr_info.set_index("id0").to_dict(orient="index")

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



for k, v in dcr_info2.items():
    v['text'] = tx1[k]['text']



all_keys = list(dcr_info2.keys())
all_texts = [v['text'] for v in dcr_info2.values()]
all_embeds_responses = oai_client.embeddings.create(input=all_texts, model='text-embedding-3-small')
all_embeds = np.stack([ee.embedding for ee in all_embeds_responses.data])

data2save = {'talk_ids': all_keys, 'embeds': all_embeds, 'talk_info': dcr_info2}

with open(here()/'data'/'interim'/'embeds_talks_dcr.pkl', 'wb') as f:
    pickle.dump(data2save, f)

with open(here()/'data'/'interim'/'talk_ids.txt', 'w') as f:
    f.writelines([f"{k}\n" for k in all_keys])

import json

with open(here()/'data'/'interim'/'talk_info.json', 'w') as f:
    json.dump(dcr_info2, f)

np.savetxt(here()/'data'/'interim'/'embeds.csv', all_embeds, delimiter=',', fmt='%0.16f')