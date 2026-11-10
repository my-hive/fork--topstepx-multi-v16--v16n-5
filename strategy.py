"""MGC reversal after an RTH sweep through prior RTH high or low."""
import numpy as np
import pandas as pd

FAMILY = "metals-stoprun-prior-rth-extremes"
PARAMS = {"rth_start": 30600, "rth_end": 53400, "entry_sec": 34200, "last_entry_sec": 46800, "exit_sec": 52200, "risk_dollars": 250}


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        target = np.zeros(len(data), dtype=np.int64)
        if symbol != "MGC":
            result[symbol] = target
            continue
        sec = data.ct_sec.to_numpy()
        session = data.session.astype(str)
        rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["rth_end"])
        rth_data = data.loc[rth]
        groups = rth_data.groupby(session[rth])
        prior_high = groups.high.max().shift(1)
        prior_low = groups.low.min().shift(1)
        highs = data.groupby(session).high.cummax()
        lows = data.groupby(session).low.cummin()
        high_level = session.map(prior_high).to_numpy()
        low_level = session.map(prior_low).to_numpy()
        swept_high = highs.to_numpy() > high_level
        swept_low = lows.to_numpy() < low_level
        active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["last_entry_sec"])
        side = np.where(swept_high, -1, np.where(swept_low, 1, 0))
        target[active] = side[active]
        result[symbol] = target
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), PARAMS["risk_dollars"], dtype=float) for symbol, data in panel.items()}
