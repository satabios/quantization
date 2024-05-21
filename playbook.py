#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from smq_quantizer import W8A8
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize


# In[2]:


model_id = "codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# In[3]:


# ############# From the previous lesson(s) of "Building your own Quantizer"
# def w8_a16_forward(weight, input, scales, bias=None):
    
#     casted_weights = weight.to(input.dtype)
#     output = F.linear(input, casted_weights) * scales
    
#     if bias is not None:
#         output = output + bias
      
#     return output

# class W8A16LinearLayer(nn.Module):
#     def __init__(self, in_features, out_features, 
#                  bias=True, dtype=torch.float32):
#         super().__init__()
        
        
#         self.register_buffer(
#             "int8_weights",
#             torch.randint(
#                 -128, 127, (out_features, in_features), dtype=torch.int8
#             )
#         )
        
#         self.register_buffer("scales", 
#                              torch.randn((out_features), dtype=dtype))
        
#         if bias:
#             self.register_buffer("bias", 
#                                  torch.randn((1, out_features), 
#                                              dtype=dtype))
        
#         else:
#             self.bias = None

#     def quantize(self, weights):
#         w_fp32 = weights.clone().to(torch.float32)

#         scales = w_fp32.abs().max(dim=-1).values / 127
#         scales = scales.to(weights.dtype)

#         int8_weights = torch.round(weights
#                         /scales.unsqueeze(1)).to(torch.int8)

#         self.int8_weights = int8_weights
#         self.scales = scales
    
#     def forward(self, input):
#         return w8_a16_forward(self.int8_weights, 
#                               input, self.scales, self.bias)


# def replace_linear_with_target_and_quantize(module, 
#                                target_class, module_name_to_exclude):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear) and not \
#         any([x == name for x in module_name_to_exclude]):
#             old_bias = child.bias
#             old_weight = child.weight

#             new_module = target_class.from_float(child, output_quant=True) #,  weight_quant="per_token", act_quant="per_token")
#             setattr(module, name, new_module)

#             # getattr(module, name).quantize(old_weight)
            
#             if old_bias is not None:
#               getattr(module, name).bias = old_bias
#         else:
#             # Recursively call the function for nested modules
#             replace_linear_with_target_and_quantize(child, 
#                      target_class, module_name_to_exclude)
# ###################################


# In[4]:


# model = AutoModelForCausalLM.from_pretrained(model_id, 
#                                     torch_dtype=torch.bfloat16, 
#                                              low_cpu_mem_usage=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)


# In[5]:


# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# In[6]:


# print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))


# In[7]:


# print("Model before:\n\n", model)


# In[8]:


# replace_linear_with_target_and_quantize(model, 
#                                         W8A16LinearLayer, ["lm_head"])

# print("Model after:\n\n", pipe.model)
# print(pipe("def hello_world():", max_new_tokens=20, 
#            do_sample=False)[0]["generated_text"])


# In[10]:


def replace_linear_with_target_and_quantize_smq(module, 
                               target_class, module_name_to_exclude):
  
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            old_bias = child.bias

            new_module = target_class.from_float(child, quantize_output=False) #,  weight_quant="per_token", act_quant="per_token")
           
            setattr(module, name, new_module)
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize_smq(child, 
                     target_class, module_name_to_exclude)

# del model
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                    torch_dtype=torch.bfloat16, 
                                             low_cpu_mem_usage=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Model:", pipe.model)


# In[11]:


replace_linear_with_target_and_quantize_smq(model, W8A8, ["lm_head"])

print("Model Post Replacement:\n\n", pipe.model)


# In[ ]:


print(pipe("def hello_world():", max_new_tokens=20, 
           do_sample=False)[0]["generated_text"])


# 
