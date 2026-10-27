# Crude oil signal (WTI front month, free data only)

A one-page Streamlit site with three boxes: **Signal** (LONG / SHORT / WAIT), **Take profit**, **Stop loss**.
It follows the front-month WTI future (Yahoo `CL=F`, the same underlying that TradingView's TVC:USOIL tracks),
refreshes every 15 seconds, and runs a stochastic multi-factor simulation (see MATHEMATICS.md).

Inputs, all free and key-less: Yahoo prices, FRED real yield, EIA Cushing stocks and refinery utilization,
RSS news headlines. Any missing input is treated as neutral.

## Run locally

    pip install -r requirements.txt
    python -m unittest test_engine.py
    streamlit run app.py

Add `?price=1` to the URL to show a small live quote in the corner.

## Deploy free

share.streamlit.io -> connect this repo -> branch `main`, file `app.py`, Python 3.12, no secrets.

## Notes

Paper-trading project. The simulation is unvalidated and makes no profitability claim.
Free quotes may be delayed by Yahoo.
