"""EIA crude release impulse: first completed 5m response, one intraday trade weekly."""
import numpy as np
import pandas as pd

FAMILY = "energy-eia-release-drift"
PARAMS = {"release_sec": 34200, "observe_seconds": 300, "hold_seconds": 3600, "risk_points": 2}


def signals(panel):
    result = {}
    for symbol, data in panel.items():
        out = np.zeros(len(data), dtype=np.int64)
        if symbol != "MCL":
            result[symbol] = out
            continue
        sec = data.ct_sec.to_numpy()
        weekday = data.index.weekday.to_numpy()
        start = PARAMS["release_sec"] + PARAMS["observe_seconds"]
        end = start + PARAMS["hold_seconds"]
        release = (weekday == 2) & (sec >= PARAMS["release_sec"]) & (sec < start)
        response = pd.Series(np.where(release, data.close - data.open, np.nan), index=data.index).groupby(data.session).sum(min_count=1)
        side = np.sign(response)
        mapped = data.session.map(side).fillna(0).to_numpy()
        active = (weekday == 2) & (sec >= start) & (sec < end)
        out[active] = mapped[active].astype(np.int64)
        result[symbol] = out
    return result


def entry_risk(panel):
    return {symbol: np.full(len(data), PARAMS["risk_points"] * 10.0, dtype=float) for symbol, data in panel.items()}
