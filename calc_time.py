import os

from config import get_args

args = get_args()

up_load_time = []
with open(f"{args.model_name}_result/_upload.txt", "r") as f:
    for line in f:
        up_load_time.append(float(line))

off_load_time = []
with open(f"{args.model_name}_result/_offload.txt", "r") as f:
    for line in f:
        off_load_time.append(float(line))

e2e_time = []
with open(f"{args.model_name}_result/_e2e.txt", "r") as f:
    for line in f:
        e2e_time.append(float(line))

print(f"Avg up load time: {sum(up_load_time) / len(e2e_time)}")
print(f"Avg off load time: {sum(off_load_time) / len(e2e_time)}")
print(f"Avg e2e time: {sum(e2e_time) / len(e2e_time)}")