import os
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="opt_6_7b")
    parser.add_argument("--hf_name", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--dataset_name", type=str, default="data_openwebtext/openwebtext_train.jsonl")

    parser.add_argument("--total_sentence_number", type=int, default=10)
    parser.add_argument("--max_input_length", type=int, default=100)
    parser.add_argument("--max_output_length", type=int, default=150)

    args = parser.parse_args()
    return args