import os
import json
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.opt.modeling_opt
from modeling_opt import OPTForCausalLM

from config import get_args

os.environ['HF_HUB_BASE_URL'] = 'https://mirror.huggingface.co/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

args = get_args()

save_name = args.model_name + "_result"
if not os.path.exists(save_name):
    os.mkdir(save_name)

data = []
with open(args.dataset_name, 'r') as f:
    for line in f:
        if line.strip() != "":
            data.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained(args.hf_name)
offload_model = OPTForCausalLM.from_pretrained(args.hf_name, device_map='auto')
origion_model = AutoModelForCausalLM.from_pretrained(args.hf_name, device_map='auto')

count = 0

tokenizer.model_max_length = args.max_input_length

pbar = tqdm(total=args.total_sentence_number)

if os.path.exists("_upload.txt"):
    os.remove("_upload.txt")

if os.path.exists("_offload.txt"):
    os.remove("_offload.txt")

if os.path.exists("_e2e.txt"):
    os.remove("_e2e.txt")

for batch in data:
    inputs = tokenizer(batch['text'],
                       return_tensors="pt",
                       truncation=True,)

    if inputs['input_ids'].shape[1] < args.max_input_length:
        continue


    inputs['input_ids'] = inputs['input_ids'].to("cuda")
    inputs['attention_mask'] = inputs['attention_mask'].to("cuda")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    origion_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=args.max_output_length,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    end.record()
    torch.cuda.synchronize()
    with open("_e2e.txt", "a") as f:
        print(start.elapsed_time(end), file=f, end='\n')

    offload_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=args.max_output_length,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    pbar.update(1)
    count += 1
    if count >= args.total_sentence_number:
        break



