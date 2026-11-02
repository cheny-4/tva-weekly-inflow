"""Verify uploaded sources, baseline LSTM initialization, nonlinear heads, and 13-week batches."""
from pathlib import Path
import importlib.util,json
import numpy as np
import torch
from utils import sha256
from model import SiteLSTM,make_batch
from train import load_data
ROOT=Path(__file__).resolve().parents[1]
for name,digest in json.loads((ROOT/'deployment_sha256.json').read_text()).items():
    assert sha256(ROOT/name)==digest,name
print('SOURCE_INTEGRITY_PASSED',flush=True)
if (ROOT/'preparation_audit.json').exists():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('baseline_model',ROOT/'references/baseline_model.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.manual_seed(42);baseline=old.SiteLSTM(1)
    torch.manual_seed(42);new=SiteLSTM(1)
    for k,v in baseline.state_dict().items():
        torch.testing.assert_close(v,new.state_dict()[k][:,:4] if k=='lstm.weight_ih_l0' else new.state_dict()[k],rtol=0,atol=0)
    assert torch.count_nonzero(new.lstm.weight_ih_l0[:,4:])==0
    assert sum(p.numel() for p in new.parameters())==74932
    a,sc,tensors,samples=load_data(ROOT/'data','cpu')
    selected=samples['train'][np.linspace(0,len(samples['train'])-1,32).astype(int)]
    x,s,y,scale,bi=make_batch(selected,tensors)
    assert x.shape==(32,13,3)
    for j,(t,b) in enumerate(selected):torch.testing.assert_close(x[j],tensors[0][t-12:t+1,b],rtol=0,atol=0)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    baseline.eval();new.eval()
    with torch.no_grad():torch.testing.assert_close(baseline(x,s[:,:1],bi),new(x,s,bi),rtol=1e-6,atol=1e-7)
    new.train();loss=(((new(x,s,bi)-y)/scale)**2).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    assert new.lstm.weight_ih_l0.grad[:,4:].abs().sum()>0,'New statics must receive learning gradients'
    print('MODEL_AND_HISTORY_CHECK_PASSED: baseline-identical LSTM initialization; 74932 parameters; 13 correct weekly steps; finite gradients',flush=True)

    from model import check_models
    check_models()
    opt=torch.optim.Adam(new.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.00001)
    used=[]
    for epoch in range(50):
        used.append(opt.param_groups[0]['lr']);opt.step();scheduler.step()
    np.testing.assert_allclose(used,.00001+.5*(.001-.00001)*(1+np.cos(np.pi*np.arange(50)/50)),rtol=1e-10)
    assert abs(opt.param_groups[0]['lr']-.00001)<1e-14
    print('COSINE_SCHEDULE_CHECK_PASSED: 50 independent-of-validation steps',flush=True)

