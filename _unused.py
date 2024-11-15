# model.generate(**tokenizer)
#
# input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")
# past_key_values = None

# for _ in range(20):
#     outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
#     next_token = outputs.logits[:, -1, :].argmax(dim=-1)
#
#     input_ids = next_token.unsqueeze(-1)
#     past_key_values = outputs.past_key_values
#
#     print(tokenizer.decode(next_token))
#
# for batch in data:
#     sentence = tokenizer(batch['text'],
#                          return_tensors='pt',
#                          truncation=True)
#     outputs = model(**sentence)
#     next_token = outputs.logits[:, -1, :].argmax(dim=-1)
#
# print("Finish")


## ---------- get kv cache from cpu ---------------
__key_cache = past_key_value[0].cpu()
__value_cahe = past_key_value[1].cpu()
start.record()

# -------------------------------------------------

## ---------------  add kv cache to cpu ---------------------------------
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
__key_cache = key_states.cpu()
__value_cache = value_states.cpu()
end.record()
torch.cuda.synchronize()
with open("_offload.txt", "a") as f:
    print(start.elapsed_time(end), file=f)
# -------------------------------------------------------------------------