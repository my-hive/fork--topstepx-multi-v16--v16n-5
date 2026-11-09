"""USDA-report volume-shock continuation in grains: noon-ET 5m volume spike (WASDE/crop reports) sets direction, hold 2h."""
import numpy as np
import pandas as pd

FAMILY = "grain-report-volume-shock"
PARAMS = {"win_start": 39600, "win_end": 39900, "hold_end": 46800, "vol_days": 40, "vol_mult": 3.0, "risk_dollars": 280}
SYMS = ("ZC", "ZS", "ZW")


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol not in SYMS:
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        sess = data.session.astype(str)
        w = (sec >= PARAMS["win_start"]) & (sec < PARAMS["win_end"])
        g = data[w].groupby(sess[w])
        vol = g.volume.sum()
        base = vol.rolling(PARAMS["vol_days"]).median().shift(1)
        shock = vol > PARAMS["vol_mult"] * base
        side = np.sign(g.close.last() - g.open.first()) * shock
        mapped = sess.map(side).fillna(0).to_numpy()
        active = (sec >= PARAMS["win_end"]) & (sec < PARAMS["hold_end"])
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), float(PARAMS["risk_dollars"])) for symbol, data in panel.items()}
