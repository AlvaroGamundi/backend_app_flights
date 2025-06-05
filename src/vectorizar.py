import torch
from transformers import BertTokenizer


class PreprocesadorBERT:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocesar(self, textos):
        input_ids = []
        attention_masks = []
        for texto in textos:
            tokens = self.tokenizer(
                texto,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(tokens['input_ids'][0])
            attention_masks.append(tokens['attention_mask'][0])
        return torch.stack(input_ids), torch.stack(attention_masks)

