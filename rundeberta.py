from transformers import AutoModelForMaskedLM
import torch
model = AutoModelForMaskedLM.from_pretrained('HannahRoseKirk/Hatemoji')
model.forward(input_ids=torch.tensor([[1,2,3]]))

