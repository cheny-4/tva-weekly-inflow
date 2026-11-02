"""Shared one-layer LSTM with independent 64 -> 16 -> 1 nonlinear heads."""
import torch
from torch import nn

class SiteLSTM(nn.Module):
    def __init__(self,layers=1,hidden=64,dropout=.2,sites=52,head_hidden=16):
        super().__init__();assert layers==1
        self.lstm=nn.LSTM(4,hidden,num_layers=1,batch_first=True)
        # All site heads begin identically, then learn independently.
        first=nn.Linear(hidden,head_hidden);last=nn.Linear(head_hidden,1)
        self.head_hidden_weight=nn.Parameter(first.weight.detach().repeat(sites,1,1))
        self.head_hidden_bias=nn.Parameter(first.bias.detach().repeat(sites,1))
        self.head_output_weight=nn.Parameter(last.weight.detach().repeat(sites,1))
        self.head_output_bias=nn.Parameter(last.bias.detach().repeat(sites))
        self.dropout=nn.Dropout(dropout)
        with torch.no_grad():
            self.lstm.bias_ih_l0[hidden:2*hidden].fill_(1.)
            self.lstm.bias_hh_l0[hidden:2*hidden].zero_()

    def forward(self,x,static,site):
        fixed=static[:,None,:].expand(-1,x.shape[1],-1)
        _,(h,_)=self.lstm(torch.cat((x,fixed),dim=-1))
        features=self.dropout(h[-1])
        local=torch.relu(torch.bmm(self.head_hidden_weight[site],features.unsqueeze(-1)).squeeze(-1)+self.head_hidden_bias[site])
        return torch.nn.functional.softplus((local*self.head_output_weight[site]).sum(-1)+self.head_output_bias[site])

def make_batch(samples,tensors,history=13):
    x,s,y,scales=tensors
    index=torch.as_tensor(samples,dtype=torch.long,device=x.device);t,b=index.T
    offsets=torch.arange(1-history,1,device=x.device)
    return x[t[:,None]+offsets,b[:,None]],s[b],y[t,b],scales[b],b

def check_models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);model=SiteLSTM()
        assert sum(p.numel() for p in model.parameters())==72884
        assert sum(p.numel() for p in model.lstm.parameters())==17920
        model.eval();x=torch.randn(6,13,3);s=torch.randn(6,1);b=torch.tensor([0,2,12,38,45,51])
        with torch.no_grad():
            before=model(x,s,b);model.head_output_bias[12]+=1;after=model(x,s,b)
            assert after[2]>before[2] and torch.equal(after[b!=12],before[b!=12])
            x2=x.clone();x2[0]+=10
            torch.testing.assert_close(model(x2,s,b)[1:],after[1:],rtol=0,atol=0)
        model.train();model(x,s,b).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('MODEL_CHECK_PASSED: 72884 parameters; independent nonlinear heads; finite gradients',flush=True)

