from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO, StringIO
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:          # the direct Yahoo endpoint is used when yfinance is missing
    yf = None

NAMES = ["price", "rate", "dollar", "inventory", "utilization",
         "crack", "freight", "curve", "surprise", "news"]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; cl-quant/2.0)"}
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


def observation(value, date, source, status="observed"):
    stamp = pd.Timestamp(date)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Invalid source number")
    return dict(value=number, date=stamp.to_pydatetime(), source=source, status=status)


def empty():
    return {name: dict(value=None, date=None, source="Unavailable",
                       status="missing") for name in NAMES}


# ------------------------------------------------------------------ prices

def _chart(symbol, interval, span):
    r = requests.get(YAHOO.format(symbol), headers=HEADERS, timeout=10,
                     params=dict(interval=interval, range=span, includePrePost="true"))
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    close = result["indicators"]["quote"][0]["close"]
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    s = pd.Series(close, index=index, dtype="float64").dropna()
    if s.empty:
        raise ValueError("No Yahoo data")
    return s.sort_index()


def _yfinance(symbol, interval, span):
    if yf is None:
        raise ValueError("yfinance unavailable")
    frame = yf.download(symbol, period=span, interval=interval, auto_adjust=False,
                        progress=False, threads=False, timeout=10)
    if frame.empty:
        raise ValueError("No Yahoo data")
    s = frame["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def yahoo(symbol, span, interval="1d"):
    try:
        return _chart(symbol, interval, span)
    except Exception:
        return _yfinance(symbol, interval, span)


def quote():
    for interval, span in (("1m", "1d"), ("5m", "5d")):
        try:
            s = yahoo("CL=F", span, interval)
            return observation(s.iloc[-1], s.index[-1], "Yahoo CL=F front month")
        except Exception:
            continue
    return empty()["price"]


# -------------------------------------------------------------------- news

BULL = [  # (pattern, weight): oil-price-bullish phrases
    (r"\b(surge|surges|surged|soar|soars|soared|jump|jumps|jumped|rally|rallies|rallied)\b", 1.0),
    (r"\b(rise|rises|rose|gain|gains|gained|climb|climbs|climbed|advance|advances|higher)\b", .6),
    (r"\b(inventor(y|ies)|stocks|stockpiles?|crude stocks)\b.{0,25}\b(draw|drawdown|fall|fell|drop|dropped|decline|declined|shrink)", 1.0),
    (r"\b(draw|drawdown)s?\b", .5),
    (r"\b(opec\+?|saudi|russia)\b.{0,30}\b(cut|cuts|curb|curbs|extend|extends|extended|reduce|reduces)\b", 1.0),
    (r"\b(production|output|supply)\b.{0,15}\b(cut|cuts|curb|curbs|disruption|disruptions|outage|outages|halt|halted)\b", 1.0),
    (r"\b(supply|supplies)\b.{0,15}\b(tight|tighten|tightens|tightening|shortage|risk|risks|fears|concerns)\b", .8),
    (r"\b(sanction|sanctions|embargo|blockade)\b", .6),
    (r"\b(attack|attacks|strike on|drone|missile|escalat\w+|tensions?|conflict|war)\b", .5),
    (r"\b(hurricane|storm|refinery (fire|outage)|pipeline (outage|shutdown))\b", .6),
    (r"\b(demand)\b.{0,15}\b(growth|rebound|recovery|strong|stronger|rise|rises|surge)\b", .8),
    (r"\b(stimulus|rate cuts?|weaker dollar|dollar (falls|weakens|slides))\b", .4),
    (r"\b(bullish|backwardation|deficit)\b", .6),
]
BEAR = [
    (r"\b(plunge|plunges|plunged|tumble|tumbles|tumbled|slump|slumps|slumped|sink|sinks|sank|crash|crashes)\b", 1.0),
    (r"\b(fall|falls|fell|drop|drops|dropped|slide|slides|slid|slip|slips|slipped|decline|declines|declined|lower|dip|dips)\b", .6),
    (r"\b(inventor(y|ies)|stocks|stockpiles?|crude stocks)\b.{0,25}\b(build|builds|rise|rose|jump|jumped|increase|increased|surge|swell)", 1.0),
    (r"\b(build|builds)\b", .4),
    (r"\b(glut|oversupply|surplus|overhang|flood\w*)\b", 1.0),
    (r"\b(opec\+?|saudi|russia)\b.{0,30}\b(hike|hikes|raise|raises|boost|boosts|increase|increases|ramp)\b", 1.0),
    (r"\b(production|output|supply)\b.{0,15}\b(rise|rises|increase|increases|growth|record|hike|hikes|boost)\b", .8),
    (r"\b(demand)\b.{0,15}\b(fears|worries|concerns|weak|weaker|weakness|slow|slows|slowdown|falls|drop|cut)\b", 1.0),
    (r"\b(recession|slowdown|tariffs?|trade war|stronger dollar|dollar (rises|strengthens|jumps|surges))\b", .6),
    (r"\b(ceasefire|cease-fire|peace|truce|talks|deal)\b", .5),
    (r"\b(bearish|contango)\b", .6),
]
NEGATION = re.compile(r"\b(no|not|without|despite|denies|denied|fails?|failed)\b")
FEEDS = [
    "https://news.google.com/rss/search?q=crude+oil+WTI+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=OPEC+oil+prices+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL=F&region=US&lang=en-US",
]


def score_headline(title):
    """Oil-price sentiment of one headline in [-1, 1] (lexicon, no paid NLP)."""
    text = title.lower()
    bull = sum(w for p, w in BULL if re.search(p, text))
    bear = sum(w for p, w in BEAR if re.search(p, text))
    raw = bull - bear
    if raw and NEGATION.search(text):
        raw *= -.5
    return math.tanh(raw / 1.5)


def news_score(items, now=None, half_life_hours=8.):
    """Recency-weighted, de-duplicated sentiment of (title, datetime) items."""
    now = now or datetime.now(timezone.utc)
    seen, total, weight = set(), 0., 0.
    for title, stamp in items:
        key = re.sub(r"\W+", " ", title.lower()).strip()[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        age = max(0., (now - stamp).total_seconds() / 3600)
        if age > 36:
            continue
        w = .5 ** (age / half_life_hours)
        total += w * score_headline(title)
        weight += w
    if weight < 1.0:                       # too little fresh news: neutral
        return 0.
    return float(np.clip(math.tanh(total / (weight + 3.)) * 2, -1, 1))


def parse_rss(xml_text):
    items = []
    root = ET.fromstring(xml_text)
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        stamp = node.findtext("pubDate")
        try:
            when = parsedate_to_datetime(stamp)
            when = when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
        except Exception:
            continue
        if title:
            items.append((title, when))
    return items


def news():
    items = []
    for url in FEEDS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            items += parse_rss(r.text)
        except Exception:
            continue
    if not items:
        raise ValueError("No news")
    now = datetime.now(timezone.utc)
    return observation(news_score(items, now), now, "RSS headline sentiment", "proxy")


# ------------------------------------------------------------ macro factors

def eia(code):
    url = f"https://www.eia.gov/dnav/pet/hist_xls/{code}.xls"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    frame = pd.read_excel(BytesIO(response.content), sheet_name=1, header=2).iloc[:, :2]
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date").sort_values("date")
    if len(frame) < 30:
        raise ValueError("Insufficient EIA history")
    return frame.set_index("date")["value"]


def public_factors():
    """Returns (inputs, daily_history, hourly_history, warnings). Every factor is optional."""
    inputs, warnings = empty(), []
    state = dict(history=[], intraday=[])
    today = datetime.now(timezone.utc).date()

    def attempt(label, fn):
        try:
            fn()
        except Exception as exc:
            warnings.append(f"{label}: unavailable ({type(exc).__name__})")

    def prices():
        s = yahoo("CL=F", "3y")
        s = s[s.index.date < today]
        state["history"] = s.astype(float).tolist()

    def hourly():
        s = yahoo("CL=F", "1mo", "1h")
        state["intraday"] = s.astype(float).tolist()[:-1]   # drop the unfinished bar

    def rates():
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "DFII10"}, headers=HEADERS, timeout=12)
        r.raise_for_status()
        s = pd.to_numeric(pd.read_csv(StringIO(r.text), index_col=0).iloc[:, 0],
                          errors="coerce").dropna()
        inputs["rate"] = observation(s.iloc[-1], s.index[-1], "FRED DFII10 real yield, percent")

    def dollars():
        s = yahoo("DX-Y.NYB", "6mo")
        s = s[s.index.date < today]
        if len(s) < 22:
            raise ValueError("Insufficient DXY history")
        inputs["dollar"] = observation(12 * np.log(s.iloc[-1] / s.iloc[-22]), s.index[-1],
                                       "Yahoo DXY; annualized 21-session log change")

    def stocks():
        s = eia("W_EPC0_SAX_YCUOK_MBBLw") / 1000
        inputs["inventory"] = observation(s.iloc[-1], s.index[-1], "EIA Cushing; million barrels")
        changes = s.diff().dropna()
        prior = changes.iloc[-27:-1]
        surprise = (changes.iloc[-1] - prior.tail(8).mean()) / max(.1, prior.std())
        inputs["surprise"] = observation(surprise, s.index[-1],
                                         "Own forecast residual; NOT analyst consensus", "proxy")

    def utilization():
        s = eia("WPULEUS3w")
        inputs["utilization"] = observation(s.iloc[-1], s.index[-1],
                                            "EIA refinery utilization, percent")

    def crack():
        aligned = pd.concat([yahoo(t, "3mo") for t in ["CL=F", "RB=F", "HO=F"]],
                            axis=1, join="inner").dropna()
        aligned = aligned[aligned.index.date < today]
        cl, rb, ho = aligned.iloc[-1]
        inputs["crack"] = observation(42 * (2 * rb + ho) / 3 - cl, aligned.index[-1],
                                      "Yahoo 3-2-1 crack, USD/bbl", "proxy")

    def headlines():
        inputs["news"] = news()

    for label, function in [("Daily CL", prices), ("Hourly CL", hourly), ("Real yield", rates),
                            ("DXY", dollars), ("Cushing", stocks), ("Utilization", utilization),
                            ("Crack", crack), ("News", headlines)]:
        attempt(label, function)
    return inputs, state["history"], state["intraday"], warnings


def demo(seed=42):
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    history = (75 + np.cumsum(rng.normal(0, 1.4, 600))).tolist()
    intraday = (history[-1] + np.cumsum(rng.normal(0, .3, 230))).tolist()
    values = [history[-1], 1.7, .03, 48, 90, 20, .4, 2, .5, 0.]
    inputs = {name: observation(v, now, "SYNTHETIC DEMO", "mock") for name, v in zip(NAMES, values)}
    return inputs, history, intraday, []
