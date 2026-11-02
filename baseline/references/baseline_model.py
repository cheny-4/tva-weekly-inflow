"""One/two stacked LSTM layers, 52 separate heads, no GNN or cross-site input."""
import torch
from torch import nn

class SiteLSTM(nn.Module):
    def __init__(self,layers=1,hidden=64,dropout=.2,sites=52):
        super().__init__();assert layers in (1,2)
        # Draw the first layer and head identically for both depths with seed42.
        base=nn.LSTM(4,hidden,num_layers=1,batch_first=True)
        head=nn.Linear(hidden,1)
        if layers==1:self.lstm=base
        else:
            self.lstm=nn.LSTM(4,hidden,num_layers=layers,batch_first=True,dropout=0.)
            with torch.no_grad():
                for name,p in base.named_parameters():getattr(self.lstm,name).copy_(p)
        self.head_weight=nn.Parameter(head.weight.detach().repeat(sites,1))
        self.head_bias=nn.Parameter(head.bias.detach().repeat(sites))
        self.dropout=nn.Dropout(dropout)
        with torch.no_grad():
            for layer in range(layers):
                getattr(self.lstm,f'bias_ih_l{layer}')[hidden:2*hidden].fill_(1.)
                getattr(self.lstm,f'bias_hh_l{layer}')[hidden:2*hidden].zero_()
    def forward(self,x,static,site):
        fixed=static[:,None,:].expand(-1,x.shape[1],-1)
        _,(h,_)=self.lstm(torch.cat((x,fixed),dim=-1))
        features=self.dropout(h[-1])
        return torch.nn.functional.softplus((features*self.head_weight[site]).sum(-1)+self.head_bias[site])

def make_batch(samples,tensors,history=13):
    x,s,y,scales=tensors
    index=torch.as_tensor(samples,dtype=torch.long,device=x.device);t,b=index.T
    offsets=torch.arange(1-history,1,device=x.device)
    return x[t[:,None]+offsets,b[:,None]],s[b],y[t,b],scales[b],b

def check_models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);one=SiteLSTM(1)
        torch.manual_seed(42);two=SiteLSTM(2)
        for k,v in one.state_dict().items():assert torch.equal(v,two.state_dict()[k]),k
        for layers,m,n in [(1,one,21300),(2,two,54580)]:
            assert sum(p.numel() for p in m.parameters())==n
            m.eval();x=torch.randn(6,13,3);s=torch.randn(6,1);b=torch.tensor([0,2,12,38,45,51])
            with torch.no_grad():
                before=m(x,s,b);m.head_bias[12]+=1;after=m(x,s,b)
                assert after[2]>before[2] and torch.equal(after[b!=12],before[b!=12])
                x2=x.clone();x2[0]+=10
                torch.testing.assert_close(m(x2,s,b)[1:],after[1:],rtol=0,atol=0)
            m.train();m(x,s,b).sum().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('MODEL_CHECK_PASSED: identical layer0/head initialization, 21300/54580 parameters, independent heads, finite gradients',flush=True)
