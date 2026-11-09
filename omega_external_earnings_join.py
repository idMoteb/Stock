#!/usr/bin/env python3
"""OMEGA external earnings catalyst research: no trading and no future disclosures."""
import os,glob,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
OUT=Path("omega_earnings_event_out");OUT.mkdir(exist_ok=True)
sources=glob.glob("oldsignals/*/ALL_SIGNALS_WITH_TARGET_CHECKS.csv")
assert len(sources)==6,sources
frames=[]
for filename in sources:
 f=pd.read_csv(filename)
 f=f.loc[f.fit100==True].copy()
 f["source_file"]=filename.split("/")[1]
 frames.append(f)
S=pd.concat(frames,ignore_index=True)
S["ticker"]=S.ticker.astype(str).str.upper().str.strip()
S["scan_date"]=pd.to_datetime(S.scan_date)
assert len(S)==20045,len(S)
print("SIGNAL_DATA",len(S),"UNIQUE_TICKERS",S.ticker.nunique(),flush=True)
file=hf_hub_download(repo_id="sovai/earnings_surprise",filename="earnings_surprise.parquet",repo_type="dataset")
schema=pq.ParquetFile(file).schema_arrow
print("EARNINGS_SCHEMA",schema.names,flush=True)
columns=["ticker","date","surprise_probability","eps_surprise","actual_earning_result","estimated_earning","date_pub"]
d=pq.read_table(file,columns=columns).to_pandas()
print("RAW_EARNINGS_ROWS",len(d),"unique",d.ticker.nunique(),flush=True)
d["ticker"]=d.ticker.astype(str).str.upper().str.strip()
d["publication_date"]=pd.to_datetime(d.date_pub,errors="coerce").dt.normalize()
d["obs_date"]=pd.to_datetime(d.date,errors="coerce").dt.normalize()
for c in ["eps_surprise","actual_earning_result","estimated_earning","surprise_probability"]:
 d[c]=pd.to_numeric(d[c],errors="coerce")
d=d[d.ticker.isin(set(S.ticker)) & d.publication_date.notna() & (d.publication_date>="2018-01-01") & (d.publication_date<="2026-11-09")]
# One release must be observable only after its publication date, not when weekly estimates appeared.
# Prevent weekly pre-event label leakage: only rows observed on/after the release are used as sources of the released EPS.
d=d[d.obs_date >= d.publication_date].copy()
print("POST_PUBLICATION_CANDIDATE_ROWS",len(d),flush=True)
# One released event per ticker, publication date, aggregate duplicate weekly records.
g=d.groupby(["ticker","publication_date"]).agg(
 eps_surprise=("eps_surprise","median"),
 actual_eps=("actual_earning_result","median"),
 estimated_eps=("estimated_earning","median"),
 obs_count=("obs_date","size"),
 first_valid_obs=("obs_date","min")
).reset_index()
g=g[np.isfinite(g.eps_surprise)&np.isfinite(g.estimated_eps)&np.isfinite(g.actual_eps)].copy()
g["eps_surprise_signed_pct"]=100*g.eps_surprise/np.maximum(np.abs(g.estimated_eps),.05)
g["eps_surprise_signed_pct"]=g.eps_surprise_signed_pct.clip(-1000,1000)
g["eps_beat"]=g.eps_surprise>0
g["publication_effective"]=g[["publication_date","first_valid_obs"]].max(axis=1)+pd.Timedelta(days=1)
g=g.sort_values(["ticker","publication_effective"])
g["prior_surprise"]=g.groupby("ticker").eps_surprise_signed_pct.shift(1)
g["prior_eps_beat"]=g.groupby("ticker").eps_beat.shift(1)
g["trailing_beats4"]=(g.groupby("ticker").eps_beat.shift(1).groupby(g.ticker).rolling(4,min_periods=1).mean().reset_index(level=0,drop=True))
g["prior_pub_date"]=g.groupby("ticker").publication_date.shift(1)
print("FINAL_PUBLISHED_EARNINGS_EVENTS",len(g),"TICKERS",g.ticker.nunique(),flush=True)
g.to_csv(OUT/"DEDUP_REAL_PUBLISHED_EARNINGS_EVENTS.csv",index=False)
l=S.sort_values(["scan_date","ticker"]).copy()
r=g.sort_values(["publication_effective","ticker"]).copy()
m=pd.merge_asof(l,r,left_on="scan_date",right_on="publication_effective",by="ticker",direction="backward",allow_exact_matches=True)
m["earnings_age_calendar_days"]=(m.scan_date-m.publication_date).dt.days
m["earnings_report_available"]=m.publication_date.notna()
m["earnings_within_7d"]=m.earnings_age_calendar_days.between(1,7)
m["earnings_within_21d"]=m.earnings_age_calendar_days.between(1,21)
m["earnings_within_45d"]=m.earnings_age_calendar_days.between(1,45)
m["positive_surprise"]=m.eps_surprise>0
m["surprise_gt_20"]=m.eps_surprise_signed_pct>=20
m["surprise_gt_50"]=m.eps_surprise_signed_pct>=50
m["surprise_gt_100"]=m.eps_surprise_signed_pct>=100
cols=["ticker","scan_date","source_file","sector","hit10","hit15","fit_score","rankpct_mio",
      "atr14_pct","return126_pct","rvol20","gain5_pct","gain10_pct","gain20_pct","gap_pct","maxdown20_pct",
      "candle_body_ratio","close_location","volatility_ratio","rank_momentum","noncross",
      "eps_surprise","actual_eps","estimated_eps","eps_surprise_signed_pct","eps_beat","prior_surprise",
      "trailing_beats4","publication_date","publication_effective","first_valid_obs",
      "earnings_age_calendar_days","earnings_report_available","earnings_within_7d",
      "earnings_within_21d","earnings_within_45d","positive_surprise","surprise_gt_20",
      "surprise_gt_50","surprise_gt_100","obs_count"]
cols=[c for c in cols if c in m.columns]
m[cols].to_csv(OUT/"SGX_ALL_20045_SIGNALS_WITH_LAGGED_EARNINGS.csv",index=False)
summary={
"source":"sovai/earnings_surprise",
"total_sgx_fit100":len(m),
"unique_sgx_stocks":S.ticker.nunique(),
"external_earnings_rows":len(d),
"published_dedup_earnings_events":len(g),
"earnings_matched_stockdays":int(m.earnings_report_available.sum()),
"earnings_7days":int(m.earnings_within_7d.sum()),
"earnings_21days":int(m.earnings_within_21d.sum()),
"earnings_45days":int(m.earnings_within_45d.sum()),
"note":"EPS actual only counted after date_pub and first observation. Conservative one-day latency. Historic EPS revisions and ticker mapping not independently audited."
}
(OUT/"MANIFEST.json").write_text(json.dumps(summary,indent=2))
print("OMEGA_EXTERNAL_EARNINGS_SUMMARY",json.dumps(summary),flush=True)
