"""
Local chrF evaluation for sonnet generation, using the provided dev set.
"""
import argparse
import torch
import sacrebleu

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT
from utils import get_device

import random

import numpy as np

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
  if torch.backends.mps.is_available():
      torch.mps.manual_seed(seed)


@torch.no_grad()
def evaluate(args):
    seed_everything(args.seed)
    device = get_device(args.use_gpu)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    prompt_dataset = SonnetsDataset(args.prompts_path)
    reference_dataset = SonnetsDataset(args.references_path)

    hypotheses = []
    references = []

    for (prompt_batch, ref_batch) in zip(prompt_dataset, reference_dataset):
        prompt_text = prompt_batch[1]
        reference_text = ref_batch[1]

        encoding = model.tokenizer(prompt_text, return_tensors='pt',
                                    padding=False, truncation=True).to(device)
        token_ids = model.generate(encoding['input_ids'],
                                    temperature=args.temperature,
                                    top_p=args.top_p)[0][0]
        generated = model.tokenizer.decode(token_ids)

        hypotheses.append(generated)
        references.append(reference_text)

        print(f"--- Sonnet ---")
        print(f"GENERATED:\n{generated}\n")
        print(f"REFERENCE:\n{reference_text}\n")

    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    print(f"\nchrF score: {chrf.score:.3f}")
    return chrf.score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompts_path", type=str,
                        default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--references_path", type=str,
                        default="data/TRUE_sonnets_held_out_dev.txt")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=11711)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluate(args)