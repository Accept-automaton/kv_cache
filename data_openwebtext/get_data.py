import os
os.environ['HF_HUB_BASE_URL'] = 'https://mirror.huggingface.co/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from tqdm import tqdm
import json

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


dataset = load_dataset("Bingsu/openwebtext_20p", split="train", streaming=True, trust_remote_code=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)

path = "openwebtext_train.jsonl"
for idx, doc in enumerate(tqdm(dataset)):
    data = {
        "text": doc["text"]
    }

    dump_jsonl([data], path, append=True)
    if idx == 5000:
        print("Collected 5000 samples")
        break
