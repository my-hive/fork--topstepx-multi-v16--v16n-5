"""Japanese-yen Monday prior-session reversal: weekly liquidity-reset mean reversion, intraday only."""
import numpy as np
import pandas as pd

FAMILY = "fx-monday-prior-session-reversal"
PARAMS = {"rth_start": 30600, "rth_end": 50400, "entry_sec": 34200, "exit_sec": 50400, "weekday": 0, "risk_dollars": 280}


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol != "6J":
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["rth_end"])
        g = data[rth].groupby(data.session[rth].astype(str))
        prior_return = (g.close.last() - g.open.first()).shift(1)
        side = -np.sign(prior_return)
        mapped = data.session.astype(str).map(side).fillna(0).to_numpy()
        active = ((data.index.weekday.to_numpy() == PARAMS["weekday"]) &
                  (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"]))
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
