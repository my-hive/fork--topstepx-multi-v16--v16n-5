"""Grain prior-session return reversal during next liquid day session."""
import numpy as np
import pandas as pd
FAMILY = "grain-prior-day-return-reversal"
PARAMS = {"entry_sec": 30600, "exit_sec": 53100, "risk_dollars": 200}
def signals(panel):
    result = {}
    for s, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if s not in ("ZC", "ZS", "ZW"):
            result[s] = out; continue
        daily = df.groupby("session", sort=False).close.last()
        direction = -np.sign(daily.diff()).shift(1)
        mapped = df.session.astype(str).map(pd.Series(direction.to_numpy(), index=daily.index.astype(str))).fillna(0).to_numpy()
        sec = df.ct_sec.to_numpy()
        out[(sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])] = mapped[(sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])].astype(np.int64)
        result[s] = out
    return result
def entry_risk(panel): return {s: np.full(len(d), PARAMS["risk_dollars"], dtype=float) for s,d in panel.items()}
