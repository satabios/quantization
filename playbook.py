import torch
import torch.nn as nn
import torch.nn.functional as F
from smooth_quant_cnn import W8A8
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "models/Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def replace_linear_with_target_and_quantize_smq(module, 
                               target_class, module_name_to_exclude):
  
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            old_bias = child.bias

            new_module = target_class.from_float(child, quantize_output=True) #,  weight_quant="per_token", act_quant="per_token")
           
            setattr(module, name, new_module)
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize_smq(child, 
                     target_class, module_name_to_exclude)

# del model
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                    # torch_dtype=torch.bfloat16,
                                    low_cpu_mem_usage=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Model:", pipe.model)


replace_linear_with_target_and_quantize_smq(model, W8A8, ["lm_head"])

print("Model Post Replacement:\n\n", pipe.model)


print(pipe("def hello_world():", max_new_tokens=20, 
           do_sample=False)[0]["generated_text"])


# 
