import os
from datetime import datetime, timezone
import streamlit as st
import data
from engine import simulate, decide, eia_events, front_expiry_days, HORIZON_HOURS

st.set_page_config(page_title="Crude Oil Signal", page_icon="🛢️", layout="wide",
                   initial_sidebar_state="collapsed")

DEMO = os.environ.get("CL_DEMO") == "1"      # offline test mode, never set in production
CAPACITY = 76.                               # Cushing working storage, million bbl
REFRESH = "15s"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@600;800&display=swap');
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
.stApp {background: radial-gradient(1200px 700px at 50% -10%, #12233b 0%, #070b12 60%) fixed; }
.block-container {padding: 0 !important; max-width: 100% !important;}
.stage {min-height: 100vh; display:flex; align-items:center; justify-content:center; padding: 4vh 5vw;}
.grid {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 28px; width:min(1280px,100%);}
@media (max-width: 900px) {.grid {grid-template-columns: 1fr; gap:18px;} .stage{align-items:flex-start;}}
.card {position:relative; border-radius: 28px; padding: 44px 32px 48px; text-align:center;
  background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.10); backdrop-filter: blur(14px);
  box-shadow: 0 30px 80px -30px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.08);
  overflow:hidden; transition: transform .35s ease, border-color .35s ease;}
.card:hover {transform: translateY(-4px);}
.card::before {content:""; position:absolute; inset:-40% -20% auto -20%; height:70%;
  background: radial-gradient(closest-side, var(--c) 0%, transparent 70%); opacity:.22; filter: blur(30px);}
.label {position:relative; font: 600 13px/1 'Inter',sans-serif; letter-spacing:.32em; text-transform:uppercase;
  color: rgba(255,255,255,.55); margin-bottom: 30px;}
.value {position:relative; font: 800 clamp(44px, 6.2vw, 88px)/1 'JetBrains Mono',monospace; color: var(--c);
  text-shadow: 0 0 40px color-mix(in srgb, var(--c) 55%, transparent); letter-spacing:-.02em;}
.long {--c:#2bf0a8;} .short {--c:#ff5577;} .wait {--c:#ffc24a;} .tp {--c:#2bf0a8;} .sl {--c:#ff5577;} .none {--c:#8794a8;}
.signal.long, .signal.short {animation: pulse 2.8s ease-in-out infinite;}
@keyframes pulse {0%,100% {box-shadow: 0 30px 80px -30px rgba(0,0,0,.8), 0 0 0 0 color-mix(in srgb, var(--c) 0%, transparent);}
  50% {box-shadow: 0 30px 80px -30px rgba(0,0,0,.8), 0 0 70px 0 color-mix(in srgb, var(--c) 35%, transparent);}}
.px {position:fixed; top:14px; right:20px; font: 600 12px 'JetBrains Mono',monospace; color: rgba(255,255,255,.35);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def factors():
    return data.public_factors()


@st.cache_data(ttl=15, show_spinner=False)
def latest_quote():
    return data.quote()


def compute():
    now = datetime.now(timezone.utc)
    if DEMO:
        inputs, history, intraday, _ = data.demo()
    else:
        inputs, history, intraday, _ = factors()
        inputs = dict(inputs)
        inputs["price"] = latest_quote()
        if not history:
            factors.clear()           # do not keep a failed download for 15 minutes
    expiry = front_expiry_days(now)
    sim = simulate(inputs, history, CAPACITY, expiry, HORIZON_HOURS, seed=42,
                   events=eia_events(now, HORIZON_HOURS), intraday=intraday)
    result = decide(inputs, history, sim, now, expiry, demo_mode=DEMO)
    result["price"] = inputs["price"]["value"]
    result["at"] = now
    return result


def card(kind, label, text, css=None):
    return f'<div class="card {kind} {css or kind}"><div class="label">{label}</div><div class="value">{text}</div></div>'


@st.fragment(run_every=REFRESH)
def dashboard():
    now = datetime.now(timezone.utc)
    try:
        result = compute()
        st.session_state["last"] = result
    except Exception:
        last = st.session_state.get("last")
        result = last if last and (now - last["at"]).total_seconds() < 300 else \
            dict(action="WAIT", tp=None, sl=None, price=None)
    action = result["action"]
    tp = f'{result["tp"]:.2f}' if result["tp"] is not None else "&mdash;"
    sl = f'{result["sl"]:.2f}' if result["sl"] is not None else "&mdash;"
    side = {"LONG": "long", "SHORT": "short"}.get(action, "wait")
    marker = ""
    if st.query_params.get("price") and result.get("price") is not None:
        marker = f'<div class="px">{result["price"]:.2f}</div>'
    st.markdown(
        marker + '<div class="stage"><div class="grid">'
        + card("signal", "Signal", action, side)
        + card("target", "Take profit", tp, "tp" if result["tp"] is not None else "none")
        + card("stop", "Stop loss", sl, "sl" if result["sl"] is not None else "none")
        + "</div></div>", unsafe_allow_html=True)


dashboard()
