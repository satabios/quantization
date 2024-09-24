import torch
from models import VGG, SimpleCNN
from Chunker import Chunker
from tqdm import tqdm
from dataset import Dataset
import copy

@torch.inference_mode()
def evaluate(
  model,
  dataloader,
  extra_preprocess = None,
        device = None
) -> float:
    model = model.to(device)
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        inputs = inputs.to(device)
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

    targets = targets.to(device)
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1)
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()



class cfg:
    model_type =  "vgg"

if(cfg.model_type == "simplecnn"):
    dataloader = Dataset('cifar10-simplecnn')
    model = SimpleCNN()
    model.load_state_dict(torch.load('data/weights/best_model.pth',map_location=torch.device('cpu'),weights_only=False))

elif(cfg.model_type == "vgg"):
    dataloader = Dataset('cifar10-vgg')
    model = VGG()#.cuda()
    checkpoint = torch.load("data/weights/vgg.cifar.pretrained.pth", weights_only=True)
    model.load_state_dict(checkpoint)


print(f"Original Model Accuracy : {evaluate(model, dataloader,device='cpu')}")


# #
# fuser = Fuse(model.eval(), dataloader)
# fused_model = fuser.fused_model.train()
# print(f"Fused Model Accuracy : {evaluate_model(fused_model, dataloader)}")
quantized_model = Chunker(copy.deepcopy(model), dataloader).model
print(quantized_model)
print(f"Original Model Accuracy : {evaluate(model, dataloader,device='cpu')}")



# def extra_preprocess(x):
#     # hint: you need to convert the original fp32 input of range (0, 1)
#     #  into int8 format of range (-128, 127)

#     return (x * 255 - 128).clamp(-128, 127).to(torch.int8)


# int8_model_accuracy = evaluate(quantized_model, dataloader,device='cuda')
# print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
