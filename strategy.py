"""Grain morning auction return continuation through midday session."""
import numpy as np
FAMILY = "grain-morning-return-continuation"
PARAMS = {"signal_sec": 34200, "exit_sec": 46800, "risk_dollars": 200}
def signals(panel):
    result={}
    for s,d in panel.items():
        out=np.zeros(len(d),dtype=np.int64)
        if s not in ('ZC','ZS','ZW','ZL'): result[s]=out; continue
        c,sec=d.close.to_numpy(),d.ct_sec.to_numpy()
        for idx in d.groupby('session',sort=False).indices.values():
            k=np.searchsorted(sec[idx],PARAMS['signal_sec'])
            if k and k<len(idx):
                side=int(np.sign(c[idx[k-1]]-c[idx[0]])); active=(sec[idx]>=PARAMS['signal_sec'])&(sec[idx]<PARAMS['exit_sec']); out[idx[active]]=side
        result[s]=out
    return result
def entry_risk(panel): return {s:np.full(len(d),PARAMS['risk_dollars'],dtype=float) for s,d in panel.items()}
