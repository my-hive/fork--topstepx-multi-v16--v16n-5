"""Rates-curve relative value: revert 60-minute ZF versus ZN return divergence, hedged."""
import numpy as np
import pandas as pd
FAMILY = "rates-curve-relative-reversion"
PARAMS = {"lookback_minutes": 60, "hold_minutes": 30, "entry_sec": 30600, "exit_sec": 50400, "risk_dollars": 200}
def signals(panel):
    result={s:np.zeros(len(d),dtype=np.int64) for s,d in panel.items()}
    if 'ZF' not in panel or 'ZN' not in panel: return result
    zf,zn=panel['ZF'],panel['ZN']
    # Exact UTC alignment only; each completed close is known at its own bar close.
    znc=zn.close.reindex(zf.index).ffill()
    zfr=zf.close.pct_change(PARAMS['lookback_minutes'])
    znr=znc.pct_change(PARAMS['lookback_minutes'])
    spread=(zfr-znr).to_numpy(); sec=zf.ct_sec.to_numpy()
    raw=-np.sign(spread); raw[~np.isfinite(raw)]=0
    held=np.zeros(len(zf),dtype=np.int64)
    for i in range(0,len(zf),PARAMS['hold_minutes']): held[i:i+PARAMS['hold_minutes']]=raw[i]
    active=(sec>=PARAMS['entry_sec'])&(sec<PARAMS['exit_sec']); held[~active]=0
    result['ZF']=held
    # opposite leg, timestamp-aligned causal reindex; ZN gets same signal at matching minute.
    temp=pd.Series(-held,index=zf.index).reindex(zn.index).ffill().fillna(0).to_numpy(dtype=np.int64)
    zsec=zn.ct_sec.to_numpy(); temp[(zsec<PARAMS['entry_sec'])|(zsec>=PARAMS['exit_sec'])]=0
    result['ZN']=temp
    return result
def entry_risk(panel): return {s:np.full(len(d),PARAMS['risk_dollars'],dtype=float) for s,d in panel.items()}
