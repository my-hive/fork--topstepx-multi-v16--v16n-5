"""Gold overnight return continuation after COMEX open, intraday only."""
import numpy as np
import pandas as pd

FAMILY = "metal-overnight-continuation"
PARAMS = {"entry_sec": 30600, "exit_sec": 50400, "overnight_start": 61200, "risk_dollars": 280}


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol != "GC":
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        # All bars after prior 17:00 and before today's 08:30; session label makes grouping causal.
        overnight = (sec >= PARAMS["overnight_start"]) | (sec < PARAMS["entry_sec"])
        g = data[overnight].groupby(data.session[overnight].astype(str))
        side = np.sign(g.close.last() - g.open.first())
        mapped = data.session.astype(str).map(side).fillna(0).to_numpy()
        active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
