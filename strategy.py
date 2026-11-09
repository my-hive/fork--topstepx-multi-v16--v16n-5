"""London-fix reversal: 10:00-11:00 CT FX move (into 16:00 London fix) reverses 11:00-12:00 CT in 6E/6B/6C basket."""
import numpy as np
import pandas as pd

FAMILY = "fx-london-fix-reversal"
PARAMS = {"obs_start": 36000, "fix_sec": 39600, "exit_sec": 43200, "atr_days": 20, "min_move": 0.5, "risk_dollars": 280}
SYMS = ("6E", "6B", "6C")


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol not in SYMS:
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        sess = data.session.astype(str)
        w = (sec >= PARAMS["obs_start"]) & (sec < PARAMS["fix_sec"])
        g = data[w].groupby(sess[w])
        move = g.close.last() - g.open.first()
        scale = move.abs().rolling(PARAMS["atr_days"]).mean().shift(1)
        side = -np.sign(move) * (move.abs() > PARAMS["min_move"] * scale)
        mapped = sess.map(side).fillna(0).to_numpy()
        active = (sec >= PARAMS["fix_sec"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
