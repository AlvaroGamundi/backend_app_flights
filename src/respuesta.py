import torch

class clase_respuesta:
    def __init__(self):
        self.clases={0:"We're happy to hear your experience was good! We hope to have you flying with us again soon.",
        1:"Thank you for your feedback! We are constantly working to improve our service.",
        2:"We're truly sorry for your experience. We will get in touch with you to improve anything that needs attention."}

    def __call__(self,tensor:torch.Tensor)-> str:
        return self.clases[tensor.item()]



