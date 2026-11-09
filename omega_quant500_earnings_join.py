#!/usr/bin/env python3
"""SGX announcement-date catalyst research; only uses events whose filing date precedes scan."""
import io, json, requests, pandas as pd,numpy as np
from pathlib import Path
OUT=Path("omega_earnings_dates_out");OUT.mkdir(exist_ok=True)
S=pd.read_csv("eventsource/SGX_ALL_20045_SIGNALS_WITH_LAGGED_EARNINGS.csv")
S["scan_date"]=pd.to_datetime(S.scan_date).astype("datetime64[ns]")
url="https://quant500.com/api/descarga/anuncios.csv"
r=requests.get(url,timeout=90,headers={"User-Agent":"OMEGA academic research / reproducible historical observation"})
print("EXTERNAL_STATUS",r.status_code,"BYTES",len(r.content), "TYPE",r.headers.get("content-type"),flush=True)
r.raise_for_status()
raw=pd.read_csv(io.BytesIO(r.content),comment="#",low_memory=False)
print("EXTERNAL_ROWS",len(raw),"COLUMNS",raw.columns.tolist(),flush=True)
if not {"ticker","announcement_date"}.issubset(raw.columns):raise RuntimeError("MISSING_NEEDED_COLUMNS")
E=raw[["ticker","announcement_date"]+([c for c in ["session","cik","source_url"] if c in raw.columns])].copy()
E["ticker"]=E.ticker.astype(str).str.upper().str.strip()
E["announcement_date"]=pd.to_datetime(E.announcement_date,errors="coerce").astype("datetime64[ns]")
E=E[E.announcement_date.notna()&E.ticker.isin(set(S.ticker))].copy()
# Conservative: event dated today cannot be used as a signal until the following calendar day, 
# irrespective of reported session/time or SEC document processing delay.
E["available_date"]=E.announcement_date+pd.Timedelta(days=1)
E=E.sort_values(["available_date","ticker"]).drop_duplicates(["ticker","available_date"])
print("MATCHABLE_EARNINGS_EVENTS",len(E),"UNIQUE_TICKERS",E.ticker.nunique(),flush=True)
L=S.sort_values(["scan_date","ticker"])
M=pd.merge_asof(L,E,left_on="scan_date",right_on="available_date",by="ticker",direction="backward")
M["earnings_event_age"]=(M.scan_date-M.announcement_date).dt.days
M["event_within_7d"]=M.earnings_event_age.between(1,7)
M["event_within_14d"]=M.earnings_event_age.between(1,14)
M["event_within_30d"]=M.earnings_event_age.between(1,30)
M["event_within_60d"]=M.earnings_event_age.between(1,60)
M.to_csv(OUT/"SGX_20045_EARNINGS_ANNOUNCEMENTS_AND_SURPRISE.csv",index=False)
for year,g in M.groupby(M.scan_date.dt.year):
 print("YEAR_MATCH",int(year),len(g),"WITHIN_7",int(g.event_within_7d.sum()),"WITHIN_30",int(g.event_within_30d.sum()),flush=True)
meta={"source":url,"events":len(E),"tickers":E.ticker.nunique(),"SGX_signals":len(M),"earnings_7d":int(M.event_within_7d.sum()),"earnings_30d":int(M.event_within_30d.sum()),"latest_source_date":str(E.announcement_date.max().date())}
(OUT/"MANIFEST.json").write_text(json.dumps(meta,indent=2))
print("EARNINGS_DATE_MANIFEST",json.dumps(meta),flush=True)
