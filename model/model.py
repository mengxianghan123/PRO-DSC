import torch.nn as nn
import torch.nn.functional as F

class PRO_DSC(nn.Module):
    def __init__(self,input_dim, hidden_dim, z_dim):
        super().__init__()
        self.pre_feature = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         )
        self.subspace = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
        self.cluster = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
    def forward(self, x):
        
        pre_feature = self.pre_feature(x)
        Z = self.subspace(pre_feature)
        logits = self.cluster(pre_feature).float()
        Z = F.normalize(Z, 2)
        logits = F.normalize(logits, 2)
        
        return Z, logits

    