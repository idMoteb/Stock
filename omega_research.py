#!/usr/bin/env python3
"""OMEGA public-data research ONLY; independent of user's journal; no orders."""
import os, json, math, re, hashlib
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pathlib import Path

START=pd.Timestamp('2026-03-02')
END=pd.Timestamp('2026-07-31')
HORIZONS=(5,10,20)
OUT=Path('omega_outputs');OUT.mkdir(exist_ok=True)
print('DOWNLOAD_BEGIN',flush=True)
price_path=hf_hub_download('AlphaDojo/dojo_stock_kline',filename='data.parquet',repo_type='dataset')
meta_path=hf_hub_download('AlphaDojo/dojo_stock_info',filename='data.parquet',repo_type='dataset')
metadata=pd.read_parquet(meta_path)
metadata['ticker']=metadata['ticker'].astype(str).str.upper().str.strip()
metadata['sector']=metadata['sector'].fillna('').astype(str).str.strip()
metadata['full_exchange_name']=metadata['full_exchange_name'].fillna('').astype(str)
meta=metadata.loc[
    metadata['market'].eq('us') & metadata['quote_type'].eq('EQUITY') &
    metadata['full_exchange_name'].str.contains('NYSE|NASDAQ',case=False,regex=True) &
    metadata['ticker'].str.fullmatch('[A-Z]{1,5}') &
    ~metadata['sector'].str.contains('crypto|digital asset',case=False,na=False)
].drop_duplicates('ticker').copy()
print('ELIGIBLE_METADATA_US_STOCKS',len(meta),flush=True)
# only necessary fields, prevent memory-heavy auxiliary columns
schema=pq.ParquetFile(price_path).schema_arrow
print('PRICE_SCHEMA_SELECTED',set(['symbol','bar_time','open','high','low','close','vol']).issubset(schema.names),flush=True)
p=pq.read_table(price_path,columns=['symbol','bar_time','open','high','low','close','vol']).to_pandas()
p=p.rename(columns={'symbol':'ticker','bar_time':'date','vol':'volume'})
p['ticker']=p.ticker.astype(str).str.upper().str.strip()
p=p[p.ticker.isin(set(meta.ticker)|{'SPY'})].copy()
p['date']=pd.to_datetime(p.date).dt.normalize()
for c in ['open','high','low','close','volume']:p[c]=pd.to_numeric(p[c],errors='coerce')
p=p[(p.date>=pd.Timestamp('2025-01-02'))&(p.date<=pd.Timestamp('2026-09-15'))]
p=p[(p.close>0)&(p.volume>=0)&(p.low<=p.close)&(p.high>=p.close)&(p.high>=p.low)].copy()
p=p.sort_values(['ticker','date']).drop_duplicates(['ticker','date'])
spy=p[p.ticker.eq('SPY')].set_index('date').close.sort_index()
print('SPY_DAILY_RECORDS',len(spy),'SPY_DATE_RANGE',spy.index.min(),spy.index.max(),flush=True)
if len(spy)<350:
    print('SPY_MISSING_DOWNLOAD_ALTERNATIVE',flush=True)
    import yfinance as yf
    q=yf.download('SPY',start='2025-01-02',end='2026-09-17',auto_adjust=True,progress=False,threads=False)
    if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
    spy=q.Close.copy();spy.index=pd.to_datetime(spy.index).normalize()
if len(spy)<350:raise RuntimeError('NO_VALID_SPY_MARKET_BENCHMARK')
all_days=spy.index
scan_days=all_days[(all_days>=START)&(all_days<=END)]
print('ACTUAL_SCAN_DATES',len(scan_days),'FIRST',scan_days.min(),'LAST',scan_days.max(),flush=True)
if len(scan_days)<60:raise RuntimeError('LESS_THAN_60_DAILY_SCANS')
# Real universe sample: >=252 historical bars and >=60 scan-period bars
sub=p[p.ticker.ne('SPY')].copy()
before=sub[sub.date<START].groupby('ticker').date.nunique()
during=sub[sub.date.between(START,END)].groupby('ticker').date.nunique()
eligible=set(before[before>=252].index)&set(during[during>=60].index)
sub=sub[sub.ticker.isin(eligible)]
meta=meta[meta.ticker.isin(eligible)]
print('STOCKS_WITH_HISTORY',len(eligible),'BARS',len(sub),flush=True)
if len(eligible)<1000:raise RuntimeError('UNIVERSE_BELOW_1000')
daily_coverage=sub[sub.date.between(START,END)].groupby('date').ticker.nunique()
minimum=min(int(daily_coverage.get(d,0)) for d in scan_days)
print('MIN_STOCK_COVERAGE_ON_EACH_DATE',minimum,flush=True)
if minimum<1000:raise RuntimeError('DAILY_QUOTES_BELOW_1000')

spy200=spy.rolling(200,min_periods=200).mean()
spy63=spy/spy.shift(63)-1
spy_bull=spy>spy200
signals=[]
for ii,(ticker,g) in enumerate(sub.groupby('ticker',sort=True)):
    if len(g)<300:continue
    # Align to actual SPY trading sessions: do NOT slide a stock across a missing market date
    g=g.set_index('date').sort_index().reindex(all_days)
    c=g.close;op=g.open;hi=g.high;lo=g.low;v=g.volume
    sma50=c.rolling(50,min_periods=50).mean()
    sma150=c.rolling(150,min_periods=150).mean()
    sma200=c.rolling(200,min_periods=200).mean()
    ema10=c.ewm(span=10,adjust=False).mean()
    ema20=c.ewm(span=20,adjust=False).mean()
    av20=v.shift(1).rolling(20,min_periods=20).mean()
    ad20=(c*v).shift(1).rolling(20,min_periods=20).mean()
    r63=c/c.shift(63)-1
    r126=c/c.shift(126)-1
    raw_liquid=(c>=10)&(av20>=1e6)&(ad20>=20e6)
    trend=(raw_liquid&(c>sma50)&(sma50>sma150)&(sma150>sma200)&(sma200>sma200.shift(20))&
           (c>=1.3*lo.shift(1).rolling(252,min_periods=252).min())&
           (c>=.75*hi.shift(1).rolling(252,min_periods=252).max())&
           ((r63-spy63)>=.05))
    width6040=(hi.shift(41).rolling(20,min_periods=20).max()-lo.shift(41).rolling(20,min_periods=20).min())
    width4020=(hi.shift(21).rolling(20,min_periods=20).max()-lo.shift(21).rolling(20,min_periods=20).min())
    width10=(hi.shift(1).rolling(10,min_periods=10).max()-lo.shift(1).rolling(10,min_periods=10).min())
    kw=(hi.shift(11).rolling(10,min_periods=10).max()-lo.shift(11).rolling(10,min_periods=10).min())
    rv=v/av20
    vcp_prep=trend&(width6040>width4020)&(width4020>width10)&(width10<=.12*c)&(v.shift(1).rolling(5,min_periods=5).mean()<=.8*av20)
    vcp=vcp_prep&(c>hi.shift(1).rolling(20,min_periods=20).max())&(c<=1.05*hi.shift(1).rolling(20,min_periods=20).max())&(rv>=1.3)&(c>op)
    wedge=raw_liquid&(c.shift(1)<=pd.concat([ema10.shift(1),ema20.shift(1)],axis=1).max(axis=1))&(c>ema10)&(c>ema20)&(c>op)&(rv>=1.2)&(kw<.18*c)
    kelltrend=raw_liquid&(ema10>ema20)&(c>sma50)&(c>ema20)&(r63-spy63>=.05)
    crossback=kelltrend&(lo<=ema20*1.01)&(c>op)&(rv>=.8)
    basebreak=kelltrend&(kw<.12*c)&(c>hi.shift(1).rolling(5,min_periods=5).max())&(rv>=1.2)
    hybrid=trend&(vcp|wedge|crossback|basebreak)
    mio=spy_bull&hybrid
    fit=15*spy_bull.astype(int)+15*raw_liquid.astype(int)+25*trend.astype(int)+30*(trend&(vcp_prep|kelltrend)).astype(int)+15*hybrid.astype(int)
    chosen=(mio&(fit==100)&c.notna() & c.index.isin(scan_days))
    if not chosen.any(): continue
    ix=np.flatnonzero(chosen.to_numpy())
    highs=hi.to_numpy(dtype=float); closes=c.to_numpy(dtype=float)
    for k in ix:
        if k+20>=len(all_days):continue
        entry=closes[k]
        future_highs=highs[k+1:k+21]
        if not np.isfinite(future_highs).all() or not np.isfinite(closes[k+20]):continue
        def hret(n):return closes[k+n]/entry-1 if np.isfinite(closes[k+n]) else np.nan
        signals.append({'ticker':ticker,'scan_date':str(all_days[k].date()),'sector':None,
          'rank_momentum':100*(.6*r63.iloc[k]+.4*r126.iloc[k]),'return126_pct':100*r126.iloc[k],
          'fit_score':int(fit.iloc[k]),'noncross':bool(vcp.iloc[k] or wedge.iloc[k] or basebreak.iloc[k]),
          'base_mio':True,'p5':hret(5),'p10':hret(10),'p20':hret(20),
          'maxhi20':float(np.max(future_highs)),'close_at_scan':float(entry),
          'hit10':bool(np.max(future_highs)>=entry*1.1),'hit15':bool(np.max(future_highs)>=entry*1.15),
          'hit10_5':bool(np.max(future_highs[:5])>=entry*1.1),
          'hit15_5':bool(np.max(future_highs[:5])>=entry*1.15),
          'hit10_10':bool(np.max(future_highs[:10])>=entry*1.1),
          'hit15_10':bool(np.max(future_highs[:10])>=entry*1.15)})
    if (ii+1)%500==0:print('PROCESSED_STOCKS',ii+1,'SIGNALS',len(signals),flush=True)
print('RAW_SIGNALS',len(signals),flush=True)
if not signals:raise RuntimeError('ZERO_CONFIRMED_LOCAL_SGX_SIGNALS')
S=pd.DataFrame(signals)
S=S.merge(meta[['ticker','sector']],on='ticker',how='left',suffixes=('','_meta'))
S['sector']=S['sector_meta'].fillna('')
S.drop(columns=['sector_meta'],inplace=True)
# Rank among full MIO candidates, not among only FIT100, requires broader set;
# HERE all signals are FIT100 so rank is a conditional rank, labeled explicitly.
S=S.sort_values(['scan_date','rank_momentum'],ascending=[True,False])
S['rank_in_fit100']=S.groupby('scan_date').cumcount()+1
S['fit100_batch_size']=S.groupby('scan_date').ticker.transform('size')
S['rankpct_fit100']=(S.fit100_batch_size-S.rank_in_fit100+.5)/S.fit100_batch_size
# Peers are other selected FIT100 stocks in same-day same-sector: proxy for full MIO scanner, not platform parity
S['peer_n']=S.groupby(['scan_date','sector']).ticker.transform('size')-1
S['daychange']=np.nan
# Need daily change to compute peer share. Rehydrate from source group fast
prev=sub.sort_values(['ticker','date']).copy()
prev['daychange']=prev.groupby('ticker').close.pct_change(fill_method=None)
lookup=prev[['ticker','date','daychange']].drop_duplicates(['ticker','date'])
lookup['scan_date']=lookup.date.dt.strftime('%Y-%m-%d')
S=S.merge(lookup[['ticker','scan_date','daychange']],on=['ticker','scan_date'],how='left',suffixes=('','_actual'))
S['daychange']=S['daychange_actual'];S.drop(columns=['daychange_actual'],inplace=True)
valid=S.sector.ne('')
g=S.groupby(['scan_date','sector'])
S['peer_pos']=g['daychange'].transform(lambda a:(a>0).sum())-S.daychange.gt(0).astype(int)
S['peer_up_frac']=S.peer_pos/S.peer_n.replace(0,np.nan)
S['a_plus']=(S.return126_pct>=140)&((S.rank_momentum>=120)|S.noncross)
S['rank975']=S.rankpct_fit100>=.975
S['csmm']=(S.return126_pct>=130)&S.noncross&(S.peer_n>=2)&(S.peer_up_frac>=.6)&valid
S['csmm_or_aplusrank']=S.csmm|(S.a_plus&S.rank975)
S['csmm_or_rank']=S.csmm|S.rank975
S['aplus_and_rank']=S.a_plus&S.rank975
models=[('SGX_FIT100_all',pd.Series(True,index=S.index)),('A_PLUS',S.a_plus),('RANK_TOP_2_5pct_FIT100_UNIVERSE',S.rank975),('CSMM',S.csmm),('CSMM_OR_AplusRank',S.csmm_or_aplusrank),('CSMM_OR_Rank',S.csmm_or_rank),('Aplus_AND_Rank',S.aplus_and_rank)]
rows=[]
for name,mask in models:
 d=S[mask];row={'model':name,'unique_stocks':int(d.ticker.nunique()),'scan_days_with_signals':int(d.scan_date.nunique()),'stock_day_signals':len(d)}
 for n in (5,10,20):
  positive=int((d[f'p{n}']>0).sum()); valid_n=int(d[f'p{n}'].notna().sum())
  row[f'positive_close_{n}d_n']=positive
  row[f'positive_close_{n}d_pct']=round(100*positive/valid_n,2) if valid_n else None
  row[f'touch_10pct_{n}d_n']=int(d[f'hit10_{n}' if n<20 else 'hit10'].sum())
  row[f'touch_10pct_{n}d_pct']=round(100*row[f'touch_10pct_{n}d_n']/len(d),2) if len(d) else None
  row[f'touch_15pct_{n}d_n']=int(d[f'hit15_{n}' if n<20 else 'hit15'].sum())
  row[f'touch_15pct_{n}d_pct']=round(100*row[f'touch_15pct_{n}d_n']/len(d),2) if len(d) else None
 rows.append(row)
T=pd.DataFrame(rows)
T.to_csv(OUT/'ONE_FINAL_RESULT_TABLE.csv',index=False)
S.to_csv(OUT/'ALL_SIGNALS_WITH_TARGET_CHECKS.csv',index=False)
dates=pd.DataFrame({'scan_date':[str(x.date()) for x in scan_days]})
dates['signals_fit100']=dates.scan_date.map(S.groupby('scan_date').size()).fillna(0).astype(int)
dates.to_csv(OUT/'ALL_DAILY_SCAN_DATES.csv',index=False)
data={'data_source':'AlphaDojo/dojo_stock_kline & dojo_stock_info','start':str(START.date()),'end':str(END.date()),
      'actual_scan_days':len(scan_days),'unique_us_stocks_with_history':len(eligible),
      'min_quoted_stocks_per_scan_day':minimum,'fit100_signals':len(S),
      'caveats':['Rank is among FIT100 signals, not all MIO signals; cannot claim MIO platform rank parity.',
                 'Sector synchronization computed among FIT100 same-sector signals; actual MIO includes non-FIT100 candidates.',
                 'Endpoint price targets use High in next 20 market sessions; closing direction close scan to close after n sessions.',
                 'Consecutive stock-day signals overlap; these are not independent trials.',
                 'Point-in-time membership and historical sector classifications not audited.',
                 'Stock prices corporate-action adjustment requires external independent audit.',
                 'No trades executed; no SL or trade execution model.']}
(OUT/'MANIFEST.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('OMEGA_MANIFEST',json.dumps(data),flush=True)
print('OMEGA_RESULTS_BEGIN',flush=True)
print(T.to_csv(index=False),flush=True)
print('OMEGA_RESULTS_END',flush=True)
