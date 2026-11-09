"""Heating-oil prior-session RTH reversal gated on expanded prior range: distillate demand shocks overreact overnight and mean-revert intraday."""
import numpy as np
import pandas as pd

FAMILY = "energy-prior-day-reversal"
PARAMS = {"rth_start": 30600, "rth_end": 54000, "entry_sec": 34200, "exit_sec": 54000, "range_lookback": 20, "range_mult": 1.2, "risk_dollars": 280}
SYMS = ("HO", "RB")


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol not in SYMS:
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        sess = data.session.astype(str)
        rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["rth_end"])
        g = data[rth].groupby(sess[rth])
        ret = g.close.last() - g.open.first()
        prior_range = (g.high.max() - g.low.min())
        base = prior_range.shift(1).rolling(PARAMS["range_lookback"]).median()
        side = -np.sign(ret.shift(1)) * (prior_range.shift(1) > PARAMS["range_mult"] * base)
        mapped = sess.map(pd.Series(side.to_numpy(), index=ret.index.astype(str))).fillna(0).to_numpy()
        active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
