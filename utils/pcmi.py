import torch 
import torch.nn.functional as F
from torch import nn

class pcmi(nn.Module):
    def __init__(self,feature_dim=128, alpha=0.3, tau=0.07):
        super().__init__()
        self.pos_mlp = nn.Linear(3, feature_dim)
        self.D = feature_dim
        self.alpha = alpha
        self.tau   = tau

    def forward(self, xyz,fr, fL):
        M, C = fr.shape
        Q, L, C2 = fL.shape
        assert C==C2, 'embed dim mismatch'

        fp = self.pos_mlp(xyz)
        fr_n = F.normalize(fr, dim=-1)   #[M,C]
        fp_n = F.normalize(fp, dim=-1)   #[M,C]
        fL_n = F.normalize(fL, dim=-1)   #[Q,L,C]

        att_L2P = torch.einsum('qlc,mc->qlm', fL_n, fr_n) / (C**0.5)  #[Q,L,M]
        att_L2P = F.softmax(att_L2P, dim=-1) 
        fpL = torch.einsum('qlm,mc->qlc', att_L2P, fp_n)              #[Q,L,C]

        key = F.normalize(fL_n + fpL, dim=-1)       #[Q,L,C]
        query = F.normalize(fr_n + fp_n, dim=-1)    #[M,C]
        score = torch.einsum('mc,qlc->qml', query, key)/(C**0.5)   #[Q,M,L)]
        att_P2L = F.softmax(score, dim=-1)      

        delta_qmc = torch.einsum('qml,qlc->qmc', att_P2L, fL)     #[Q,M,C]
        fr_hat_qmc = fr[None, :, :] + self.alpha * delta_qmc      #[Q,M,C]    

        m_vis = torch.einsum('qmc,qlc->qml',fr_hat_qmc,fL_n).sum(dim=-1)   #[Q,M]
        return fr_hat_qmc, m_vis