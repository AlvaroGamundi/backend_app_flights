import torch
from transformers import BertForSequenceClassification

#prueba cambios


class ModeloBERT:
    def __init__(self, path_modelo, num_labels=3, device='cpu'):
        self.device = torch.device(device)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model.load_state_dict(torch.load(path_modelo, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predecir(self, input_ids, attention_masks):
        self.model.eval()
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
            predicciones = torch.argmax(outputs.logits, dim=1)
        return predicciones.cpu()

    