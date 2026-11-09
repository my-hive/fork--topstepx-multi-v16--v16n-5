"""Energy overnight-shock continuation: overnight move > k x 20d mean RTH range keeps going to 13:00 CT (RB, NG, HO)."""
import numpy as np
import pandas as pd

FAMILY = "energy-overnight-shock-continuation"
PARAMS = {"entry_sec": 30600, "trade_sec": 34200, "exit_sec": 46800, "overnight_start": 61200, "atr_days": 20, "shock_mult": 1.5, "risk_dollars": 280}
SYMS = ("RB", "HO")


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol not in SYMS:
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        sess = data.session.astype(str)
        rth = (sec >= PARAMS["entry_sec"]) & (sec < 50400)
        g = data[rth].groupby(sess[rth])
        atr = (g.high.max() - g.low.min()).rolling(PARAMS["atr_days"]).mean().shift(1)
        overnight = (sec >= PARAMS["overnight_start"]) | (sec < PARAMS["entry_sec"])
        go = data[overnight].groupby(sess[overnight])
        move = go.close.last() - go.open.first()
        side = np.sign(move) * (move.abs() > PARAMS["shock_mult"] * atr.reindex(move.index))
        mapped = sess.map(side).fillna(0).to_numpy()
        active = (sec >= PARAMS["trade_sec"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
