import torch
import numpy as np
import sys
import os

from model import Model
from dataset import DatasetTrain, DatasetValid
from bpemb import BPEmb

def preprocess_text(text: str):
    data = text.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
    data = data.split()
    data = ' '.join(data)

    return data

def predict_next_words(model, text):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = BPEmb(lang='bn', vs=5000, dim=300)
  text = preprocess_text(text)
  text = tokenizer.encode_ids(text)

  text = torch.Tensor([text[-3:]]).long()
  state_h, state_c = model.init_state(3)

  with torch.no_grad():
    y_pred, (state_h, state_c) = model(text.long().to(device), (state_h.float().to(device), state_c.float().to(device)))

  preds = torch.argmax(y_pred).item()
  print(preds)
  predicted_word = ""
  predicted_word = tokenizer.decode_ids([preds])
  #print(predicted_word)
  return predicted_word

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(embedding_dim = 128, hidden_dim = 1000)
    model.load_state_dict(torch.load('./saved_model/best_model.pt'))
    model.to(device)
    model.eval()
    while(True):
        text = input("Enter your line: ")
        
        if text == "0":
            print("Execution completed.....")
            break
        
        else:
            #predict_next_words(model, text)
            result = predict_next_words(model, text)
            print(result)

    #print(summary(model))
