import torch
import numpy as np
import requests
import torch.nn as nn
from torch.nn import functional as F

############################################################

torch.manual_seed(1337)

block_size = 256      ## max content length for predictions
batch_size = 32 
max_iters  = 50000
eval_interval = 5000
learning_rate = 3e-4             ## 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
vocab_size = 1000
n_embd  = 384                  ## every id gets embedded to vector of this size
n_head  = 6
n_layer = 6
dropout = 0.2

############################################################
PATH = 'MortyGPT.pt'
model = BigramLanguageModel()
model.load_state_dict(torch.load(PATH))
model.eval()

m = model.to(device)



## Kick off generation with some starting token. In this case id 0

context = torch.zeros(  (1, 1),  dtype=torch.long, device=device   )

gen_text = m.generate(context, max_new_tokens=500)[0].tolist()

print(  decode(gen_text)   )
