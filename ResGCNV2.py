import torch
from torch import nn

# Model Extended
class ResGCNV2(nn.Module):
    def __init__(self, previous_model) -> None:
        super(ResGCNV2, self).__init__()
        self.model = previous_model
        self.fcn = nn.Linear(128, 3, bias=True)
    
    def forward(self, x):
        embedding, feature = self.model(x)
        # print("Embedding: ", embedding.shape)
        out = self.fcn(embedding)
        # print("Out: ", out.shape)
        return out, embedding

