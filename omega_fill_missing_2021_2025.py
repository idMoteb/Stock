#!/usr/bin/env python3
"""OMEGA public-data research ONLY; independent of user's journal; no orders."""
import os, json, math, re, hashlib
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pathlib import Path

START=pd.Timestamp(os.environ['STUDY_START'])
END=pd.Timestamp(os.environ['STUDY_END'])
HORIZONS=(5,10,20)
OUT=Path('omega_new_gap_'+os.environ['STUDY_START']+'_'+os.environ['STUDY_END']);OUT.mkdir(exist_ok=True)
print('DOWNLOAD_BEGIN',flush=True)
# Independent provider, pre-2026; four large shards; no user-supplied trade journal.
price_paths=[hf_hub_download('HexQuant/Stocks-Daily-Price',filename=f'data/train-{i:05d}-of-00004.parquet',repo_type='dataset') for i in range(4)]
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
pieces=[]
for pi,path in enumerate(price_paths):
    q=pq.read_table(path,columns=['symbol','date','open','high','low','close','volume']).to_pandas()
    q=q[(q.date >= (START-pd.Timedelta(days=620)).strftime('%Y-%m-%d')) & (q.date <= (END+pd.Timedelta(days=50)).strftime('%Y-%m-%d'))]
    pieces.append(q)
    print('SOURCE_SHARD_LOADED',pi,len(q),flush=True)
p=pd.concat(pieces,ignore_index=True)
del pieces
p=p.rename(columns={'symbol':'ticker'})
p['ticker']=p.ticker.astype(str).str.upper().str.strip()
p=p[p.ticker.isin(set(meta.ticker)|{'SPY'})].copy()
p['date']=pd.to_datetime(p.date).dt.normalize()
for c in ['open','high','low','close','volume']:p[c]=pd.to_numeric(p[c],errors='coerce')
p=p[(p.date>=START-pd.Timedelta(days=620))&(p.date<=END+pd.Timedelta(days=50))]
p=p[(p.close>0)&(p.volume>=0)&(p.low<=p.close)&(p.high>=p.close)&(p.high>=p.low)].copy()
p=p.sort_values(['ticker','date']).drop_duplicates(['ticker','date'])
spy=p[p.ticker.eq('SPY')].set_index('date').close.sort_index()
print('SPY_DAILY_RECORDS',len(spy),'SPY_DATE_RANGE',spy.index.min(),spy.index.max(),flush=True)
if len(spy)<len(scan_days)+190:
    print('SPY_MISSING_DOWNLOAD_ALTERNATIVE',flush=True)
    import yfinance as yf
    q=yf.download('SPY',start=str((START-pd.Timedelta(days=620)).date()),end=str((END+pd.Timedelta(days=50)).date()),auto_adjust=True,progress=False,threads=False)
    if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
    spy=q.Close.copy();spy.index=pd.to_datetime(spy.index).normalize()
if len(spy)<len(scan_days)+190:raise RuntimeError('NO_VALID_SPY_MARKET_BENCHMARK')
all_days=spy.index
scan_days=all_days[(all_days>=START)&(all_days<=END)]
print('ACTUAL_SCAN_DATES',len(scan_days),'FIRST',scan_days.min(),'LAST',scan_days.max(),flush=True)
if len(scan_days)<6:raise RuntimeError('LESS_THAN_SIX_DAILY_SCANS')
# Real universe sample: >=252 historical bars and >=60 scan-period bars
sub=p[p.ticker.ne('SPY')].copy()
before=sub[sub.date<START].groupby('ticker').date.nunique()
during=sub[sub.date.between(START,END)].groupby('ticker').date.nunique()
eligible=set(before[before>=252].index)&set(during[during>=max(5,min(60,len(scan_days)//2))].index)
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
    chosen=(mio&c.notna() & c.index.isin(scan_days))
    if not chosen.any(): continue
    # PRE-SIGNAL SCIENCE METRICS: all trailing-only at scan-day close
    one_ret=c.pct_change(fill_method=None)
    true_range=pd.concat([hi-lo,(hi-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    atr14=true_range.rolling(14,min_periods=14).mean()/c
    vol20=one_ret.rolling(20,min_periods=20).std()
    vol60=one_ret.rolling(60,min_periods=60).std()
    vol_ratio=vol20/vol60
    volume_ratio=v/v.shift(1).rolling(20,min_periods=20).mean()
    gain20=(c/c.shift(20)-1)
    gain5=(c/c.shift(5)-1)
    gain10=(c/c.shift(10)-1)
    efficiency20=(c-c.shift(20)).abs()/(c.diff().abs().rolling(20,min_periods=20).sum())
    closeloc=(c-lo)/(hi-lo).replace(0,np.nan)
    candle_body=(c-op)/(hi-lo).replace(0,np.nan)
    upper_wick=(hi-pd.concat([op,c],axis=1).max(axis=1))/(hi-lo).replace(0,np.nan)
    previous_20_high=hi.shift(1).rolling(20,min_periods=20).max()
    previous_52high=hi.shift(1).rolling(252,min_periods=252).max()
    previous_20_low=lo.shift(1).rolling(20,min_periods=20).min()
    break20=(c/previous_20_high-1)
    at_52high=c/previous_52high
    range20=hi.rolling(20,min_periods=20).max()-lo.rolling(20,min_periods=20).min()
    prior_range20=(hi.shift(20).rolling(20,min_periods=20).max()-lo.shift(20).rolling(20,min_periods=20).min())
    rangecompression=range20/prior_range20
    gap=(op/c.shift(1)-1)
    updays10=(one_ret.gt(0)).rolling(10,min_periods=10).sum()
    maxdown20=one_ret.rolling(20,min_periods=20).min()
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
          'atr14_pct':float(100*atr14.iloc[k]),'vol20_pct':float(100*vol20.iloc[k]),'vol60_pct':float(100*vol60.iloc[k]),
          'volatility_ratio':float(vol_ratio.iloc[k]),'rvol20':float(volume_ratio.iloc[k]),
          'gain5_pct':float(100*gain5.iloc[k]),'gain10_pct':float(100*gain10.iloc[k]),'gain20_pct':float(100*gain20.iloc[k]),
          'efficiency20':float(efficiency20.iloc[k]),'close_location':float(closeloc.iloc[k]),
          'candle_body_ratio':float(candle_body.iloc[k]),'upper_wick_ratio':float(upper_wick.iloc[k]),
          'break20_pct':float(100*break20.iloc[k]),'near52high_pct':float(100*at_52high.iloc[k]),
          'rangecompression20':float(rangecompression.iloc[k]),'gap_pct':float(100*gap.iloc[k]),
          'updays10':float(updays10.iloc[k]),'maxdown20_pct':float(100*maxdown20.iloc[k]),
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
# Rank among ALL local-MIO screened candidates for this scan date, prior to FIT100 filtering.
S=S.sort_values(['scan_date','rank_momentum'],ascending=[True,False])
S['mio_rank']=S.groupby('scan_date').cumcount()+1
S['mio_batch_size']=S.groupby('scan_date').ticker.transform('size')
S['rankpct_mio']=(S.mio_batch_size-S.mio_rank+.5)/S.mio_batch_size
S['fit100']=S.fit_score.eq(100)
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
S['a_plus']=S.fit100&(S.return126_pct>=140)&((S.rank_momentum>=120)|S.noncross)
S['rank975']=S.fit100&(S.rankpct_mio>=.975)
S['csmm']=S.fit100&(S.return126_pct>=130)&S.noncross&(S.peer_n>=2)&(S.peer_up_frac>=.6)&valid
S['csmm_or_aplusrank']=S.csmm|(S.a_plus&S.rank975)
S['csmm_or_rank']=S.csmm|S.rank975
S['aplus_and_rank']=S.a_plus&S.rank975
models=[('MIO_local_screen_all',pd.Series(True,index=S.index)),('SGX_FIT100_all',S.fit100),('A_PLUS',S.a_plus),('RANK_TOP_2_5pct_FIT100_UNIVERSE',S.rank975),('CSMM',S.csmm),('CSMM_OR_AplusRank',S.csmm_or_aplusrank),('CSMM_OR_Rank',S.csmm_or_rank),('Aplus_AND_Rank',S.aplus_and_rank)]
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
data={'data_source':'HexQuant/Stocks-Daily-Price OHLCV (independent older years) + AlphaDojo current company metadata','start':str(START.date()),'end':str(END.date()),
      'actual_scan_days':len(scan_days),'unique_us_stocks_with_history':len(eligible),
      'min_quoted_stocks_per_scan_day':minimum,'mio_signals_total':len(S),'fit100_signals':int(S.fit100.sum()),
      'caveats':['Rank is among all local MIO screened candidates; vendor and exchange can differ from actual MIO platform.',
                 'Sector synchronization computed among all local-MIO same-sector candidates, not official MIO platform feed.',
                 'Endpoint price targets use High in next 20 market sessions; closing direction close scan to close after n sessions.',
                 'Consecutive stock-day signals overlap; these are not independent trials.',
                 'Point-in-time membership and historical sector classifications not audited.',
                 'Stock prices corporate-action adjustment requires external independent audit.',
                 'No trades executed; no SL or trade execution model.', 'Independent stock OHLCV archive differs from AlphaDojo 2026 by price vendor and adjustment convention.']}
(OUT/'MANIFEST.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('OMEGA_MANIFEST',json.dumps(data),flush=True)
print('OMEGA_RESULTS_BEGIN',flush=True)
print(T.to_csv(index=False),flush=True)
print('OMEGA_RESULTS_END',flush=True)
