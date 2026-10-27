import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

YEAR_HOURS = 252 * 23
NY = ZoneInfo("America/New_York")
COST = 0.04            # assumed round-trip cost, USD/bbl
HORIZON_HOURS = 23.0   # one trading session
TICK = 0.01
KT = (0.5, 0.75, 1.0)             # take-profit distances, in units of horizon sigma
KS = (0.5, 0.75, 1.0)         # stop-loss distances, in units of horizon sigma
MIN_Z = 1.28                  # one-sided ~90% on the held-out half
MIN_EV = 0.02                 # minimum expected R-multiple on the held-out half


def number(inputs, key, default=0.0):
    item = inputs.get(key, {}).get("value") if isinstance(inputs.get(key), dict) else None
    return default if item is None else float(item)


def penalty(stock, capacity, expiry):
    u = np.asarray(stock) / capacity
    soft = np.logaddexp(0, 30 * (u - .85)) / 30
    raw = 60 * np.exp(-expiry / 30) * soft / np.maximum(1 - u, .02)
    return 500 * np.tanh(raw / 500)


# ---------------------------------------------------------------- calendar

def market_open(at):
    local = at.astimezone(NY)
    d, h = local.weekday(), local.hour
    return (d == 6 and h >= 18) or (d < 4 and h != 17) or (d == 4 and h < 17)


def trading_hours(start, end):
    total = 0.
    while start < end:
        stop = min(end, start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        if market_open(start):
            total += (stop - start).total_seconds() / 3600
        start = stop
    return total


def eia_events(now, horizon):
    """Trading-hour offsets of scheduled EIA releases (Wed 10:30 New York)."""
    local = now.astimezone(NY)
    release = (local + timedelta(days=(2 - local.weekday()) % 7)).replace(
        hour=10, minute=30, second=0, microsecond=0)
    if release <= local:
        release += timedelta(days=7)
    first = trading_hours(now, release)
    return [first + 115 * k for k in range(6) if 0 < first + 115 * k <= horizon]


def _bdays_back(d, n):
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def cl_expiry(year, month):
    """Last trade date of the CL contract expiring in the given calendar month.
    Three business days before the 25th (or before the business day preceding
    the 25th if the 25th is not a business day). Exchange holidays ignored."""
    d = date(year, month, 25)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return _bdays_back(d, 3)


def front_expiry_days(now):
    today = now.astimezone(NY).date()
    y, m = today.year, today.month
    for _ in range(3):
        exp = cl_expiry(y, m)
        if exp >= today:
            return float((exp - today).days)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return 30.


# ---------------------------------------------------------------- simulation

def robust_sigma(series, annualization, window):
    x = np.asarray(series[-(window + 1):], float)
    d = np.diff(x)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return None
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826
    if mad > 0:
        d = np.clip(d, med - 5 * mad, med + 5 * mad)   # tames contract-roll gaps
    s = float(np.std(d, ddof=1) * math.sqrt(annualization))
    return s if np.isfinite(s) and s > 0 else None


def estimate_sigma(history, intraday):
    daily = robust_sigma(history, 252, 60) if len(history) >= 30 else None
    hourly = robust_sigma(intraday, YEAR_HOURS, 230) if intraday is not None and len(intraday) >= 60 else None
    parts = [s for s in (daily, hourly) if s is not None]
    if not parts:
        return 25.
    return float(np.clip(math.sqrt(np.mean([s * s for s in parts])), 1, 120))


def simulate(inputs, history, capacity=76., expiry=20., hours=HORIZON_HOURS,
             paths=4000, seed=42, events=(), intraday=None):
    spot = number(inputs, "price", float("nan"))
    if inputs["price"]["value"] is None or not np.isfinite(spot):
        raise ValueError("Current price unavailable")
    if not 0 < capacity or expiry < 0:
        raise ValueError("Invalid capacity or expiry")
    stock = number(inputs, "inventory", capacity * .6)
    stock = min(max(stock, 0.), capacity)
    sigma = estimate_sigma(history, intraday)
    anchor = float(np.median(history[-60:])) if len(history) else spot
    rng = np.random.default_rng(seed)
    steps = max(24, min(1000, int(hours * 2)))
    grid = np.linspace(0, hours, steps + 1)
    dt = hours / YEAR_HOURS / steps

    # 5-session time-series momentum, normalised by horizon volatility
    if len(history) >= 6:
        move = spot - float(history[-5])
        trend = math.tanh(move / (sigma * math.sqrt(5 / 252)))
    else:
        trend = 0.
    trend_drift = .5 * sigma * trend       # USD/bbl/year (Sharpe-0.5 style)

    price = np.full(paths, spot)
    variance = np.full(paths, sigma * sigma)
    inventory = np.full(paths, stock)
    samples = np.empty((steps + 1, paths), np.float32)
    variance_mean = np.empty(steps + 1)
    inventory_bands = np.empty((steps + 1, 3))
    samples[0] = price
    variance_mean[0] = variance.mean()
    inventory_bands[0] = np.quantile(inventory, [.05, .5, .95])
    target = np.clip(sigma * sigma * np.exp(
        .25 * np.tanh(number(inputs, "freight")) +
        .2 * np.tanh(number(inputs, "curve") / 10)), 1, 120 ** 2)
    multiplier = np.exp(
        .35 * np.tanh((number(inputs, "crack", 15) - 15) / 10) +
        .25 * np.tanh((number(inputs, "utilization", 88) - 88) / 5))
    macro = -4 * (number(inputs, "rate", 1.5) - 1.5) - 25 * number(inputs, "dollar")
    news = float(np.clip(number(inputs, "news"), -1, 1))
    event_steps = [int(np.searchsorted(grid, t)) for t in events if 0 < t <= hours]
    cap_count = 0
    for j in range(1, steps + 1):
        zp, independent, zi = rng.standard_normal((3, paths))
        zv = -.45 * zp + math.sqrt(1 - .45 ** 2) * independent
        decay = 2 ** (-grid[j] / 46)
        response = -3 * math.tanh(number(inputs, "surprise")) * decay
        sentiment = 40 * news * decay
        drift = (anchor - price + macro + 12 * (multiplier - 1) + response + sentiment
                 + trend_drift - penalty(inventory, capacity, expiry))
        counts = rng.poisson(4 * dt, paths)
        price += drift * dt + np.sqrt(variance * dt) * zp
        price += 3 * np.sqrt(counts) * rng.standard_normal(paths)
        for _ in range(event_steps.count(j)):
            price += rng.standard_normal(paths)
        candidate = variance + 5 * (target - variance) * dt + 12 * np.sqrt(variance * dt) * zv
        cap_count += np.count_nonzero(candidate > 120 ** 2)
        variance = np.clip(candidate, 1, 120 ** 2)
        inventory += 4 * math.sqrt(dt) * zi
        inventory = capacity - np.abs(np.mod(inventory, 2 * capacity) - capacity)
        samples[j] = price
        variance_mean[j] = variance.mean()
        inventory_bands[j] = np.quantile(inventory, [.05, .5, .95])
    if not np.isfinite(samples).all():
        raise ValueError("Numerical instability")
    return dict(grid=grid, samples=samples, variance=variance_mean,
                inventory=inventory_bands,
                bands=np.quantile(samples, [.05, .5, .95], axis=1).T,
                sigma=float(sigma), cap_fraction=float(cap_count / (steps * paths)),
                dt_year=float(dt), hours=float(hours), trend=float(trend), news=news)


# ---------------------------------------------------------------- decision

def barrier_pnl(samples, spot, direction, tp, sl, shift, cost=COST):
    """Per-path P&L of a bracket order (TP/SL) opened at spot.
    Discrete monitoring is corrected with the Broadie-Glasserman-Kou shift.
    A path touching both barriers on the same step is booked as a stop-out.
    Unfilled paths are closed at the horizon."""
    move = direction * (samples[1:].astype(np.float64) - spot)
    hit_tp = move >= tp - shift
    hit_sl = move <= -(sl - shift)
    n = move.shape[0]
    first_tp = np.where(hit_tp.any(axis=0), hit_tp.argmax(axis=0), n + 1)
    first_sl = np.where(hit_sl.any(axis=0), hit_sl.argmax(axis=0), n + 1)
    pnl = np.where(first_sl <= first_tp, np.where(first_sl <= n, -sl, move[-1]),
                   tp)
    pnl = np.where((first_tp > n) & (first_sl > n), np.clip(move[-1], -sl, tp), pnl)
    return pnl - cost, first_tp, first_sl


def _score(pnl):
    return float(pnl.mean()), float(pnl.std(ddof=1) / math.sqrt(len(pnl)))


def decide(inputs, history, sim, now, expiry_days, demo_mode=False):
    """Return dict(action in LONG/SHORT/WAIT, tp, sl, ...)."""
    out = dict(action="WAIT", tp=None, sl=None, ev=0., z=0., p_tp=0., reason="")
    price = inputs["price"]
    if price["value"] is None:
        out["reason"] = "no quote"
        return out
    if not demo_mode:
        if not market_open(now):
            out["reason"] = "market closed"
            return out
        if price["date"] is not None and (now - price["date"]).total_seconds() > 1800:
            out["reason"] = "stale quote"
            return out
    if len(history) < 60:
        out["reason"] = "insufficient history"
        return out
    if sim["cap_fraction"] > .01:
        out["reason"] = "numerical variance cap"
        return out
    if expiry_days <= 1:
        out["reason"] = "contract roll"
        return out

    spot = float(price["value"])
    samples = sim["samples"]
    half = samples.shape[1] // 2
    train, valid = samples[:, :half], samples[:, half:]
    horizon_sigma = sim["sigma"] * math.sqrt(sim["hours"] / YEAR_HOURS)
    step_sigma = math.sqrt(float(np.mean(sim["variance"])) * sim["dt_year"])
    shift = 0.5826 * step_sigma
    floor = 0.15

    best = None
    for direction in (1, -1):
        for kt in KT:
            for ks in KS:
                tp = max(floor, kt * horizon_sigma)
                sl = max(floor, ks * horizon_sigma)
                pnl, _, _ = barrier_pnl(train, spot, direction, tp, sl, shift)
                sd = float(pnl.std(ddof=1))
                key = float(pnl.mean()) / sd if sd > 0 else -1e9   # per-trade Sharpe: avoids lottery-style brackets
                if best is None or key > best[0]:
                    best = (key, direction, tp, sl)
    _, direction, tp, sl = best
    pnl, first_tp, first_sl = barrier_pnl(valid, spot, direction, tp, sl, shift)
    mean, se = _score(pnl)
    z = mean / se if se > 0 else 0.
    ev_r = mean / sl
    p_tp = float(np.mean(first_tp < first_sl))
    out.update(ev=ev_r, z=z, p_tp=p_tp)
    if ev_r >= MIN_EV and z >= MIN_Z:
        out["action"] = "LONG" if direction == 1 else "SHORT"
        out["tp"] = round(spot + direction * tp, 2)
        out["sl"] = round(spot - direction * sl, 2)
        out["reason"] = "positive validated expectancy"
    else:
        out["reason"] = "no validated edge"
    return out
