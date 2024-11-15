import os
import json

from tqdm import tqdm
from transformers import AutoTokenizer
import transformers.models.opt.modeling_opt
from modeling_opt import OPTForCausalLM

os.environ['HF_HUB_BASE_URL'] = 'https://mirror.huggingface.co/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = "opt_350m"
hf_name = "facebook/opt-350m"
dataset_name = 'data_openwebtext/openwebtext_train.jsonl'

save_name = model_name + "_result"
if not os.path.exists(save_name):
    os.mkdir(save_name)

data = []
with open(dataset_name, 'r') as f:
    for line in f:
        if line.strip() != "":
            data.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained(hf_name)
model = OPTForCausalLM.from_pretrained(hf_name, device_map='auto')

count = 0

max_input_length = 10
max_output_length = 20
tokenizer.model_max_length = max_input_length

for batch in data:
    inputs = tokenizer(batch['text'],
                       return_tensors="pt",
                       truncation=True,)

    if inputs['input_ids'].shape[1] < max_input_length:
        continue

    inputs['input_ids'] = inputs['input_ids'].to("cuda")
    inputs['attention_mask'] = inputs['attention_mask'].to("cuda")

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output_length,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True
    )


    # 输出生成的文本
    generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
    print(generated_text)
    break


