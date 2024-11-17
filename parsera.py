import argparse
from transformers import AutoModel, AutoTokenizer
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')

args = parser.parse_args()

model=AutoModel.from_pretrained(args.model)

print(model)