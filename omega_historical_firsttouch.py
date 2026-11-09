"""Event-based SGX research ONLY. Next-session open entry, 20 sessions maximum.
Conservative same-day stop-first collision, no lookahead for signals. Zero trades sent."""
import os, pathlib, numpy as np,pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
P=pathlib.Path("omega_historical_firsttouch_results");P.mkdir(exist_ok=True)
files=["downloaded/old/ALL_SIGNALS_WITH_TARGET_CHECKS.csv","downloaded/y2023/ALL_SIGNALS_WITH_TARGET_CHECKS.csv"]
S=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
S=S[S.fit100.eq(True)].drop_duplicates(["ticker","scan_date"]).copy()
assert len(S)==9443,len(S)
print("SIGNAL_SOURCE",len(S),"UNIQUE",S.ticker.nunique(),flush=True)
allp=[]
for fidx in range(4):
    fh=hf_hub_download("HexQuant/Stocks-Daily-Price",filename=f"data/train-{fidx:05d}-of-00004.parquet",repo_type="dataset")
    q=pq.read_table(fh,columns=["symbol","date","open","high","low","close"]).to_pandas().rename(columns={"symbol":"ticker"})
    q=q[(q.date>="2022-01-01") & (q.date<="2025-12-17") & (q.ticker.isin(set(S.ticker)))].copy()
    allp.append(q)
    print("HIST_PRICE_SHARD",fidx,len(q),flush=True)
p=pd.concat(allp,ignore_index=True)
del allp
p=p[p.ticker.isin(set(S.ticker)|{"SPY"})].copy()
p["date"]=pd.to_datetime(p.date).dt.normalize()
p=p.sort_values(["ticker","date"]).drop_duplicates(["ticker","date"])
import exchange_calendars as xcals
cal=xcals.get_calendar("XNYS")
spy=pd.DatetimeIndex(cal.sessions_in_range("2022-01-03","2025-12-17")).tz_localize(None).tolist()
dayidx={pd.Timestamp(x):i for i,x in enumerate(spy)}
p["idx"]=p.date.map(dayidx)
p=p[p.idx.notna()].copy()
p["idx"]=p.idx.astype(int)
PAYS={t:g.set_index("idx")[["open","high","low","close"]].sort_index() for t,g in p.groupby("ticker")}
print("REPLAY_PRICES",len(p),"STOCKS",len(PAYS),flush=True)

def leg(entry,arr,target,stop=0.08):
    target_price=entry*(1+target)
    stop_price=entry*(1-stop)
    anycollision=False
    for n,(_,r) in enumerate(arr.iterrows(),1):
        o,h,l,c=map(float,(r.open,r.high,r.low,r.close))
        if o<=stop_price:
            return (o/entry-1)*100,n,"GAP_STOP",False
        if l<=stop_price and h>=target_price:
            return -stop*100,n,"BOTH_STOP_FIRST",True
        if l<=stop_price:
            return -stop*100,n,"STOP",False
        if o>=target_price:
            return target*100,n,"GAP_TARGET_CAPPED",False
        if h>=target_price:
            return target*100,n,"TARGET",False
    return (float(arr.iloc[-1].close)/entry-1)*100,20,"TIME",False
rows=[]
for j,r in enumerate(S.itertuples(index=False)):
    d=pd.Timestamp(r.scan_date);k=dayidx.get(d)
    if k is None or k+20>=len(spy):continue
    g=PAYS.get(r.ticker)
    if g is None:continue
    ix=list(range(k+1,k+21))
    if not set(ix).issubset(g.index):continue
    arr=g.loc[ix]
    try:entry=float(arr.iloc[0].open)
    except:continue
    if not np.isfinite(entry) or entry<=0:continue
    a=leg(entry,arr,.10);b=leg(entry,arr,.15)
    rows.append({"ticker":r.ticker,"scan_date":str(d.date()),"entry_next_open":round(entry,6),
         "touch10_scan_close":bool(r.hit10),"touch15_scan_close":bool(r.hit15),
         "p20_close_to_close_pct":float(r.p20)*100,
         "gross_split_pct":.5*(a[0]+b[0]),
         "net_split_pct":.5*(a[0]+b[0])-0.20,
         "net_only15_pct":b[0]-0.20,
         "leg10_exit_day":a[1],"leg10_exit_reason":a[2],
         "leg15_exit_day":b[1],"leg15_exit_reason":b[2],
         "stop_first_collision":bool(a[3] or b[3]),
         "entry_to_20close_pct":100*(float(arr.iloc[-1].close)/entry-1),
         "worst_day_low_from_entry_pct":100*(arr.low.min()/entry-1),
         "best_day_high_from_entry_pct":100*(arr.high.max()/entry-1)})
    if (j+1)%1000==0:print("EVENTS",j+1,"SIM",len(rows),flush=True)
T=pd.DataFrame(rows)
T.to_csv(P/"ALL_2023_2025_SIGNALS_FIRST_TOUCH_NEXT_OPEN_10_15_SPLIT.csv",index=False)
print("SIM_DONE",len(T),"wins",int((T.net_split_pct>0).sum()),"hit15",int(T.touch15_scan_close.sum()),"ambiguity",int(T.stop_first_collision.sum()),flush=True)
print("NET_SPLIT_MEAN",T.net_split_pct.mean(),"median",T.net_split_pct.median(),"pwin",100*(T.net_split_pct>0).mean(),flush=True)
print("NET_ONLY15_MEAN",T.net_only15_pct.mean(),flush=True)
