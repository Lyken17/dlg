#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer
import torch
from transformers.models.distilbert.modeling_distilbert import Transformer
from transformers import AutoModelWithLMHead, AutoTokenizer


import torch.nn.functional as F
from torch.autograd import grad


# In[2]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."


# In[6]:


input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]


# In[7]:

bert = model.distilbert
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# In[16]:

model = model.to(device)
input = input.to(device)
trans = bert.transformer
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


# In[17]:

dlbrt_output = bert.transformer(
    x=inputs_embeds,
    attn_mask=attention_mask,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)


# In[18]:


dlbrt_output


# In[19]:


from transformers.activations import gelu

hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
prediction_logits = model.vocab_transform(hidden_states)  # (bs, seq_length, dim)
prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
prediction_logits = model.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
prediction_logits = model.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

labels = torch.randint(0, 28996, (30, )).to(device)
loss = F.cross_entropy(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

# In[20]:


plist = []
for k, v in model.distilbert.transformer.named_parameters():
    print(k, v.size())
    plist.append(v)


# In[21]:


trans = bert.transformer
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


# In[22]:


dlbrt_output = bert.transformer(
    x=inputs_embeds,
    attn_mask=attention_mask,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)

hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
# prediction_logits = model.vocab_transform(hidden_states)  # (bs, seq_length, dim)
# prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
# prediction_logits = model.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
# prediction_logits = model.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)


# In[23]:

dy_dx = grad(hidden_states.sum(), bert.transformer.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))


# In[27]:


bert.transformer.zero_grad()

dummy_embeds = torch.randn(inputs_embeds.size(), device=device).requires_grad_(True)
optimizer = torch.optim.Adam([dummy_embeds, ], lr=1e-2)


for i in range(300000):
    optimizer.zero_grad()
    bert.transformer.zero_grad()

    dlbrt_output = bert.transformer(
        x=dummy_embeds,
        attn_mask=attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = dlbrt_output[0]
    dummy_dy_dx = grad(hidden_states.sum(), bert.transformer.parameters(), create_graph=True)

    grad_diff = 0
    grad_count = 0
    for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
        grad_diff += ((gx - gy) ** 2).sum()
        grad_count += gx.nelement()
    grad_diff.backward()
    if i % 100 == 0:
        print(i, "%.2f" % grad_diff.item())
    optimizer.step()

# In[28]:

# This can be obtained in another example
seq_length = tokenizer.encode(sequence, return_tensors="pt").view(-1).size(0) 

position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
position_ids = position_ids.unsqueeze(0).expand_as(input).to(device)
position_embeds = bert.embeddings.position_embeddings(position_ids)

word_embeddings = dummy_embeds - position_embeds

# reverse query the token IDs
token_ids = []
for idx in range(seq_length):
    # temp = bert.embeddings.word_embeddings(input[:, idx:idx+1]).view(-1)
    temp = word_embeddings[:, idx:idx+1].view(-1)
    distance = ((bert.embeddings.word_embeddings.weight - temp) ** 2).sum(dim=1)
    token_id = distance.argmin().item()
    token_ids.append(token_id)
    
print(tokenizer.decode(token_ids))