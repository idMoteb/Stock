#!/usr/bin/env python3
"""Cross-era never-wait-to-sell take-profit audit. Research ONLY; no live trades.
Exit instantly on first available daily HIGH crossing +10%, +15%; conservative STOP-FIRST
if high and low cross both in same daily candle. No future data on signal date."""
import os, json, pathlib, numpy as np, pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import exchange_calendars as xcals
ERA=os.environ.get("OMEGA_ERA","old")
OUT=pathlib.Path("omega_extended_firsttouch_"+ERA);OUT.mkdir(exist_ok=True)
FILES={"old":["downloaded/y2021/ALL_SIGNALS_WITH_TARGET_CHECKS.csv","downloaded/y2023/ALL_SIGNALS_WITH_TARGET_CHECKS.csv","downloaded/y2024/ALL_SIGNALS_WITH_TARGET_CHECKS.csv"],
       "new":["downloaded/jan/ALL_SIGNALS_WITH_TARGET_CHECKS.csv","downloaded/mar/ALL_SIGNALS_WITH_TARGET_CHECKS.csv","downloaded/aug/ALL_SIGNALS_WITH_TARGET_CHECKS.csv"]}
S=pd.concat([pd.read_csv(f) for f in FILES[ERA]],ignore_index=True)
S=S[S.fit100.eq(True)].drop_duplicates(["ticker","scan_date"]).copy()
print("SIGNAL_COUNT",ERA,len(S),"tickers",S.ticker.nunique(),flush=True)
tickers=set(S.ticker)
allp=[]
if ERA=="old":
  for i in range(4):
    p=hf_hub_download("HexQuant/Stocks-Daily-Price",filename=f"data/train-{i:05d}-of-00004.parquet",repo_type="dataset")
    q=pq.read_table(p,columns=["symbol","date","open","high","low","close"]).to_pandas().rename(columns={"symbol":"ticker"})
    q=q[(q.date>="2020-01-01") & (q.date<="2025-12-17") & q.ticker.isin(tickers)]
    allp.append(q)
    print("SHARD",i,len(q),flush=True)
  calendar_end="2025-12-17"
  start="2020-01-02"
else:
  p=hf_hub_download("AlphaDojo/dojo_stock_kline",filename="data.parquet",repo_type="dataset")
  q=pq.read_table(p,columns=["symbol","bar_time","open","high","low","close"]).to_pandas().rename(columns={"symbol":"ticker","bar_time":"date"})
  q=q[(q.date>="2025-01-01") & (q.date<="2026-11-06") & q.ticker.isin(tickers)]
  allp.append(q)
  calendar_end="2026-11-06"
  start="2025-01-02"
P=pd.concat(allp,ignore_index=True)
del allp
P["date"]=pd.to_datetime(P.date).dt.normalize()
for c in ["open","high","low","close"]:P[c]=pd.to_numeric(P[c],errors="coerce")
P=P.sort_values(["ticker","date"]).drop_duplicates(["ticker","date"])
calendar=xcals.get_calendar("XNYS")
days=pd.DatetimeIndex(calendar.sessions_in_range(start,calendar_end)).tz_localize(None)
di={pd.Timestamp(t):j for j,t in enumerate(days)}
P["k"]=P.date.map(di)
P=P[P.k.notna()].copy();P.k=P.k.astype(int)
BARS={t:g.set_index("k")[["open","high","low","close"]].sort_index() for t,g in P.groupby("ticker")}
print("HIST_BARS",len(P),"TICKERS",len(BARS),"TRADING_DAYS",len(days),flush=True)
horizons=[5,10,20,40,60]
stops=[0.05,0.08,0.12,None]
def leg(entry,O,H,L,C,n,target,stop):
  barrier=entry*(1+target)
  tg=np.flatnonzero(H>=barrier)
  a=int(tg[0]) if len(tg) else 10**9
  if stop is not None:
    sbar=entry*(1-stop)
    st=np.flatnonzero(L<=sbar)
    b=int(st[0]) if len(st) else 10**9
  else:b=10**9
  if b<=a and b<n:
    if O[b]<=sbar:return (O[b]/entry-1)*100,b+1,"GAP_STOP"
    if a==b:return -100*stop,b+1,"BOTH_STOP_FIRST"
    return -100*stop,b+1,"STOP"
  if a<n:return target*100,a+1,"TARGET"
  return (C[n-1]/entry-1)*100,n,"TIME"
rows=[];missing=0
for i,r in enumerate(S.itertuples(index=False)):
  key=pd.Timestamp(r.scan_date);k=di.get(key)
  b=BARS.get(r.ticker)
  if k is None or b is None or (k+1) not in b.index:
    missing+=1;continue
  a=b.loc[b.index>=k+1]
  idx=a.index.to_numpy()
  if not len(idx) or idx[0]!=k+1:missing+=1;continue
  # STOP at first missing trading session: later values may not be backfilled.
  contiguous=np.flatnonzero(idx!=np.arange(k+1,k+1+len(idx)))
  count=min(60,int(contiguous[0]) if len(contiguous) else len(idx))
  if count<1:missing+=1;continue
  ar=a.iloc[:count].to_numpy(dtype=float)
  if not np.isfinite(ar).all() or np.any(ar[:,0]<=0):missing+=1;continue
  op,hi,lo,cl=ar.T
  entry=float(op[0]);snclose=float(getattr(r,"close_at_scan"))
  for n in horizons:
    if count<n:continue
    # Each target can trigger as early as the very FIRST subsequent exchange session.
    for stop in stops:
      x=leg(entry,op,hi,lo,cl,n,.10,stop)
      y=leg(entry,op,hi,lo,cl,n,.15,stop)
      ret=(x[0]+y[0])/2-.20
      hh=100*(np.max(hi[:n])/snclose-1)
      rows.append({"ticker":r.ticker,"scan_date":r.scan_date,"horizon_sessions":n,
        "stop_pct":100*stop if stop is not None else -1,
        "entry_next_open":entry,"scan_close":snclose,"touch15_vs_scan_close":int(hh>=15),
        "touch15_vs_next_open":int(np.max(hi[:n])>=entry*1.15),
        "touch10_vs_next_open":int(np.max(hi[:n])>=entry*1.10),
        "high_n_from_scan_pct":hh,
        "net_split_pct":ret,"net_profitable":int(ret>0),
        "leg10_exit_day":x[1],"leg15_exit_day":y[1],
        "leg10_reason":x[2],"leg15_reason":y[2],
        "both_targets_reached":int(x[2]=="TARGET" and y[2]=="TARGET"),
        "both_exit_same_day":int(x[1]==y[1]),
        "stop_leg15":int("STOP" in y[2]),
        "max_adverse_low_pct":100*(np.min(lo[:n])/entry-1),
        "entry_to_exit20_pct":100*(cl[n-1]/entry-1)})
  if i and i%1000==0: print("COMPLETED",i,"scenario_rows",len(rows),flush=True)
R=pd.DataFrame(rows)
assert len(R)>50000,(len(R),missing)
R.to_csv(OUT/"ALL_FIRST_TOUCH_TARGETS_5_10_20_40_60_STOP_5_8_12_NONE.csv.gz",index=False,compression="gzip")
summary=R.groupby(["horizon_sessions","stop_pct"]).agg(n=("ticker","size"),touch15=("touch15_vs_scan_close","mean"),net=("net_split_pct","mean"),net_winners=("net_profitable","mean"),stop15=("stop_leg15","mean"),both_targets=("both_targets_reached","mean")).reset_index()
summary.to_csv(OUT/"BASELINE_CROSS_HORIZON_STOP_COMPARISON.csv",index=False)
summary["stop_pct"]=summary.stop_pct.replace(-1,np.nan)
print("SUMMARY",summary.to_string(index=False),flush=True)
M={"era":ERA,"source":"HexQuant 2020-2025" if ERA=="old" else "AlphaDojo 2025-2026","observed_end":calendar_end,
    "signal_count":len(S),"source_rows":len(P),"valid_scenario_rows":len(R),
    "missing_signals":missing,"maturity_not_a_loss":True,
    "early_exit":"first touch of +10% and +15% separately from next-session open, regardless of what session they occur on",
    "stop_first":"if daily OHLC crosses stop and target in same session, assume stop filled before target",
    "cost_pct":.20,"horizons":horizons,"stops":[5,8,12,"none"],"research_only":True,
    "notes":"Previous 20-session outcomes can be reproduced as horizon20 stop8; longer time windows only count when completely matured, not forced losses."}
(OUT/"MANIFEST.json").write_text(json.dumps(M,indent=2))
print("OUTPUT_MANIFEST",json.dumps(M),flush=True)
