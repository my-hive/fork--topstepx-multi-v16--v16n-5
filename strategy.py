"""First-hour (08:30-09:30 CT) direction continues through the session in ZL and HE (ag/livestock open-auction flow)."""
import numpy as np
import pandas as pd

FAMILY = "ag-first-hour-continuation"
PARAMS = {"fh_start": 30600, "fh_end": 34200, "exit_sec": 50400, "risk_dollars": 280}
SYMS = ("ZL", "HE")


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol not in SYMS:
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        fh = (sec >= PARAMS["fh_start"]) & (sec < PARAMS["fh_end"])
        g = data[fh].groupby(data.session[fh].astype(str))
        side = np.sign(g.close.last() - g.open.first())
        mapped = data.session.astype(str).map(side).fillna(0).to_numpy()
        active = (sec >= PARAMS["fh_end"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
