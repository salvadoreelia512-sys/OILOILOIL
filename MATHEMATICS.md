# Arithmetic WTI multi-factor research model

Time is trading years: 252 sessions times 23 hours per year.
Price P is USD/barrel and may be negative.
Inventory I and assumed working capacity C are million barrels.
V is arithmetic variance, USD^2/barrel^2/year.

## Price system

dP = [kappa*(anchor-P) + M + betaQ*(F-1) - B(I,tau) + R + S(t) + m]dt
     + sqrt(V)dW_P + J_g dN_g + sum_k J_E,k delta_Tk(dt)

M = -beta_r*(real_yield-1.5) - beta_D*dollar_change

S(t) = 40*news*2^(-t/46)                     news in [-1,1], decays over 46 trading hours
m    = 0.5*sigma0*tanh((P0-P_5d)/(sigma0*sqrt(5/252)))   5-session time-series momentum

news = recency-weighted (8h half-life), de-duplicated lexicon sentiment of free
Google News / Yahoo Finance RSS headlines. Missing factors are neutral (0), never blocking.

crack = 42*(2*RBOB + ULSD)/3 - CL

F = exp(0.35*tanh((crack-15)/10)
        + 0.25*tanh((refinery_utilization-88)/5))

u = I/C
softplus_k(x) = log(1+exp(k*x))/k

B = 500*tanh(
    60*exp(-tau/30)*softplus_30(u-0.85)
    / (500*max(1-u,0.02))
)

tau is the user-supplied remaining expiry time, held fixed in a snapshot.
This is a storage-pressure hypothesis, not a calibrated reconstruction of 2020.
Available storage, lease commitments, delivery access and roll mechanics are absent.

## Variance and inventory

Theta = clip(
    sigma0^2 * exp(0.25*tanh(freight_z)+0.2*tanh(curve_slope/10)),
    1, 120^2
)

dV = 5*(Theta-V)dt + 12*sqrt(V)dW_V

corr(dW_P,dW_V) = -0.45

dI = 4*dW_I + reflection at 0 and C

sigma0 is estimated from 60 completed daily arithmetic price differences.
Continuous-contract roll changes can distort this estimate.
Positive annual dollar curve slope means contango.
Freight is an observed standardized VLCC rate, not an invented shipping proxy.

## Jumps and EIA

Geopolitical events follow Poisson intensity 4/year.
Their independent jump sizes are Normal(0, 3^2) USD/barrel.
Conditional on n events, total jump size is 3*sqrt(n)*Z.

Future EIA releases use scheduled Normal(0,1) jumps.
Default schedule: Wednesday 10:30 America/New_York, with regular weekly spacing.
Holiday exceptions are not supplied by an external calendar.

Latest EIA surprise is an own-forecast residual using preceding weekly changes,
not paid analyst consensus. Its assumed drift response decays over 46 trading hours.

## Numerical method

Euler-Maruyama uses correlated normal draws.
Variance is clipped to [1,120^2] as a numerical safeguard.
Inventory overshoots are reflected; prices are never floored at zero.
Frequent upper variance clipping blocks public signals.
The snapshot seed makes the random draws reproducible.

## Decision rule (Long / Short / Wait) and TP/SL

Horizon is one trading session (23 trading hours); 4000 paths, fixed seed 42.
sigma0 blends robust daily (60d) and hourly (230 bars) arithmetic volatility; increments
beyond 5 MAD are clipped so contract-roll gaps do not inflate it.

For every direction d in {long, short} and bracket (TP, SL) = (k_t, k_s) * sigma0*sqrt(T),
k_t in {0.5, 0.75, 1}, k_s in {0.5, 0.75, 1}, the simulated paths give a bracket-order P&L:
TP hit first -> +TP, SL hit first (or same step) -> -SL, neither -> marked at the horizon,
minus an assumed $0.04/bbl round-trip cost. Discrete monitoring is corrected with the
Broadie-Glasserman-Kou barrier shift 0.5826*sigma*sqrt(dt).

Selection and validation use disjoint halves of the paths: the bracket with the best
per-trade Sharpe on the first half is re-evaluated on the second half. A signal is issued
only if validated expectancy >= 0.02 SL-units and its one-sided z >= 1.28. Otherwise WAIT.
This stops Monte Carlo noise from creating trades.

WAIT is also forced when the market is closed, the quote is older than 30 minutes, fewer
than 60 completed daily bars exist, variance clipping is active, or the front contract is
within one day of expiry (roll risk). Expiry is computed from the CL rule (3 business days
before the 25th of the preceding month; exchange holidays ignored).

TP and SL are quote -/+ the selected distances, rounded to $0.01.

These are model outputs, not a validated trading edge. Coefficients are unvalidated and
no backtest is included.
