#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer
import torch
from transformers.models.distilbert.modeling_distilbert import Transformer
from transformers import AutoModelWithLMHead, AutoTokenizer


import torch.nn.functional as F
from torch.autograd import grad


# In[144]:
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

bert = model.distilbert

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = model.to(device)
trans = bert.transformer

optim = torch.optim.SGD(bert.parameters(), lr=0.1)


# In[149]:


# You may replace it with your own sentences here.
sequence = f"The Conference and Workshop on Neural Information Processing Systems is a machine learning and computational neuroscience conference held every December."
input = tokenizer.encode(sequence, return_tensors="pt")

input = input.to(device)
input_shape = input.size()
inputs_embeds = bert.embeddings(input)
attention_mask = torch.ones(input_shape, device=device)
head_mask = None
head_mask = bert.get_head_mask(head_mask, bert.config.num_hidden_layers)

output_attentions = bert.config.output_attentions
output_hidden_states = (
    bert.config.output_hidden_states
)
return_dict =  bert.config.use_return_dict

optim.zero_grad()
dlbrt_output = bert.transformer(
    x=inputs_embeds,
    attn_mask=attention_mask,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)

hidden_states = dlbrt_output[0] 
hidden_states.sum().backward() # can be replaced by any other loss functions


# In[155]:


leaked_seq_length = bert.embeddings.position_embeddings.weight.grad.var(dim=-1).nonzero().max()
leaked_token_ids = bert.embeddings.word_embeddings.weight.grad.var(dim=-1).nonzero().view(-1)
leaked_words = tokenizer.decode(leaked_token_ids) # The order is not preserved.


# In[157]:


s1 = set(leaked_token_ids.tolist())
s2 = set(tokenizer.encode(sequence))
diff = s1 - s2
if len(diff) == 0:
    print("All words have been leaked.")
print("--" * 40)

leaked_words = []
for i in range(leaked_token_ids.size(0)):
    leaked_word = tokenizer.decode(leaked_token_ids[i])
    leaked_words.append(leaked_word)
print("Leaked sentence length:", leaked_seq_length.item() + 1)
print("Leaked words:", "|".join(leaked_words))
print("--" * 40)

origin_tokens = tokenizer.encode(sequence, return_tensors="pt").view(-1)
original_words = []
for i in range(origin_tokens.size(0)):
    original_word = tokenizer.decode(origin_tokens[i])
    original_words.append(original_word)
print("Original sentence length:", len(original_words))
print("Original sentence:", "|".join(original_words))
