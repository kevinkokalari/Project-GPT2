'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F


from torch import device, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluate_paraphrase import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

### To save progress:
import os
from glob import glob
# for hyperparameter search
import optuna 

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    # not used in hidden token version: 
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    self.linear = args.linear 

    if args.linear:
      for param in self.gpt.parameters():
        param.requires_grad = False
      for param in self.paraphrase_detection_head.parameters(): 
        # when training linear model, still need parameters in linear layer
        param.requires_grad = True
    else: # defult full fine-tuning
      for param in self.gpt.parameters():
        param.requires_grad = True 
    

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE

    # Input sentences through GPT-2 model to get hidden state of final token. 
    gptOutputs = self.gpt(input_ids, attention_mask)
    lastToken = gptOutputs['last_token']

    if self.linear:
      logits = self.paraphrase_detection_head(lastToken)
      
    else:       
      logits = self.gpt.hidden_state_to_token(lastToken)
      # Take out results of id:s 8505 and 3919 (index"yes"=1 and "no"=0)
      logits = logits[:, [3919, 8505]]

    return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")




def load_part_trained(model_name, model, optimizer): # added function
  """
  Function that lets us load a partly trained model (after each epoch)
  (inspired by https://medium.com/data-science/training-language-models-on-google-colab-6e145ff092bf)
  """
  
  checkpoints = glob('/content/drive/MyDrive/checkpoint_epoch_*.pt')
  if not checkpoints:
    checkpoints = glob('checkpoint_epoch_*.pt')
  if checkpoints:
    latest = max(checkpoints, key=lambda f: int(f.split('_epoch_')[1].replace('.pt', '')))
    checkpoint = torch.load(latest, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_dev_acc = checkpoint['best_dev_acc']
    dev_f1 = checkpoint['dev_f1']
    print(f"Checkpoint loaded from epoch {epoch+1}")
    
    return epoch+1, best_dev_acc
  else:
    print(f"No checkpoint found, starting from epoch 1.")
    return 0, 0

    

def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)
  # These two lines for small testing! Remove when training full data
  #para_train_data = para_train_data[:500]
  #para_dev_data = para_dev_data[:100]
  if args.subset: # smaller subset surning hyperparametersearch
    para_train_data = para_train_data[:args.subset]
    para_dev_data   = para_dev_data[:args.subset//4]

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn, 
                                     # added:
                                     num_workers=2,       # background loading 
                                     pin_memory=True      # faster CPU to GPU transfer
                                     )
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn,
                                    # added:
                                    num_workers=2,       # background loading 
                                    pin_memory=True      # faster CPU to GPU transfer
                                    )

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  


  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  model_name = "paraphraseModel"

  if not args.search: 
    start_epoch, best_dev_acc = load_part_trained(model_name, model, optimizer)
  else:
    start_epoch = 0 
    best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(start_epoch, args.epochs): # starting from epoch loaded in checkpoint
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)


      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch+1}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
    
    if not args.search: 
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,'best_dev_acc': best_dev_acc, 'dev_f1': dev_f1
      }, f'checkpoint_epoch_{epoch+1}.pt')  # separate file per epoch
      os.system(f'cp checkpoint_epoch_{epoch+1}.pt /content/drive/MyDrive/') # copy to Google Drive for backup
  return best_dev_acc # for hyper-parameter search
    



@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False) # added weights_only=False to load the full model state dict

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=para_dev_data.collate_fn,
                                  # added:
                                  num_workers=2,       # background loading 
                                  pin_memory=True      # faster CPU to GPU transfer
                                  )
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn,
                                    # added:
                                    num_workers=2,       # background loading 
                                    pin_memory=True      # faster CPU to GPU transfer
                                    )

  # Correct
  dev_para_acc, dev_para_f1, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}, dev paraphrase f1 :: {dev_para_f1 :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  

  parser.add_argument("--linear", action='store_true') # default full fine-tuning
  
  # hyperparameter search arguments:
  parser.add_argument("--subset", type=int, default=None) # if none, use full data (subset for hyperparameter search)
  parser.add_argument("--search", action='store_true')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args



def objective(trial):  
    args = get_args()
    args.search = True
    args.subset = 1000  # smaller subset for the search
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    args.epochs = trial.suggest_categorical("epochs", [3, 5, 7])
    args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    args = add_arguments(args)
    args.filepath = f"trial_{trial.number}.pt"

    result = train(args) 

    for f in glob('checkpoint_epoch_*.pt'):
        os.remove(f)

    return result # returns best validation accuracy (what we want to maxmize)



if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  
  if args.search:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(study.best_params)
  else:
    train(args)
    test(args)
 





