import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import time

# ── 섹터별 종목 리스트 ────────────────────────────────────────────────────────
SECTORS = {
    '1': ('🔬 반도체',
          ['NVDA','AVGO','QCOM','TXN','AMD','INTC','AMAT','KLAC',
           'LRCX','MCHP','MU','ASML','ARM','ADI','MRVL','SNDK','WDC','STX']),
    '2': ('💻 소프트웨어',
          ['MSFT','ORCL','CRM','ADBE','NOW','WDAY','INTU','TEAM',
           'HUBS','PLTR','PANW','SHOP','SNOW','DDOG','ZS','CRWD']),
    '3': ('🏥 헬스케어 / 바이오',
          ['AMGN','GILD','VRTX','REGN','BIIB','MRNA','ILMN','INCY',
           'ALNY','BMRN','ISRG','LLY','ABBV','MRK','PFE','BMY']),
    '4': ('📱 빅테크 / 인터넷',
          ['AAPL','GOOGL','GOOG','AMZN','META','NFLX','TSLA',
           'UBER','LYFT','SNAP','PINS','MTCH']),
    '5': ('⚡ 에너지',
          ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO',
           'OXY','HAL','DVN','APA']),
    '6': ('🏦 금융',
          ['JPM','BAC','WFC','C','GS','MS','BLK','AXP',
           'V','MA','PYPL','SQ']),
    '7': ('🛒 나스닥 상위 50 (전체)',
          []),   # fdr로 동적 로드
}

print("=" * 55)
print("  🇺🇸  미국 주식 섹터별 저평가 종목 분석기")
print("=" * 55)
for key, (name, _) in SECTORS.items():
    print(f"  {key}. {name}")
print("=" * 55)

while True:
    choice = input("분석할 섹터 번호를 입력하세요: ").strip()
    if choice in SECTORS:
        break
    print("  ⚠️  1~7 중 하나를 입력해주세요.")

sector_name, tickers = SECTORS[choice]

if choice == '7':
    print(f"\n{sector_name} 리스트 로딩 중...")
    try:
        nasdaq_list = fdr.StockListing('NASDAQ')
        tickers = nasdaq_list['Symbol'].head(50).tolist()
    except Exception as e:
        print(f"리스트 불러오기 실패: {e}")
        tickers = []
else:
    print(f"\n{sector_name} 종목 {len(tickers)}개를 분석합니다.")

valid_stocks = []
if len(tickers) > 0:
    print("재무 데이터 수집을 시작합니다. 잠시만 기다려주세요!\n")

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            print(f"⚠️ {ticker}: 데이터 응답 없음 (에러 처리)")
            continue

        roe = info.get('returnOnEquity', None)
        roe = roe * 100 if roe is not None else None

        per = info.get('trailingPE', None)
        psr = info.get('priceToSalesTrailing12Months', None)
        pbr = info.get('priceToBook', None)

        if None in [roe, per, psr, pbr] or per <= 0 or psr <= 0 or pbr <= 0:
            print(f"⚠️ {ticker}: 재무제표 불량 또는 누락 (에러 처리)")
            continue

        # 모멘텀: 6개월·12개월 수익률 (주가 이력 추가 호출)
        mom_6m = mom_12m = None
        try:
            hist = stock.history(period='1y')
            if not hist.empty:
                cur = hist['Close'].iloc[-1]
                p6  = hist['Close'].iloc[-126] if len(hist) > 126 else hist['Close'].iloc[0]
                p12 = hist['Close'].iloc[0]
                mom_6m  = round((cur - p6)  / p6  * 100, 2)
                mom_12m = round((cur - p12) / p12 * 100, 2)
        except Exception:
            pass

        valid_stocks.append({
            '종목코드': ticker,
            '기업명':   info.get('shortName', ticker),
            'ROE':      round(roe, 2),
            'PER':      round(per, 2),
            'PSR':      round(psr, 2),
            'PBR':      round(pbr, 2),
            'MOM_6M':   mom_6m,
            'MOM_12M':  mom_12m,
        })
        print(f"✅ {ticker} 데이터 수집 완료")

    except Exception as e:
        print(f"🚨 {ticker}: 시스템 에러 발생 (제외)")
        continue
    time.sleep(0.5)

if not valid_stocks:
    print("유효한 데이터를 가진 종목이 없습니다.")
else:
    df = pd.DataFrame(valid_stocks)

    # 모멘텀 컬럼 NaN 처리 (못 받은 종목)
    df['MOM_6M']  = pd.to_numeric(df['MOM_6M'],  errors='coerce')
    df['MOM_12M'] = pd.to_numeric(df['MOM_12M'], errors='coerce')
    has_mom = df['MOM_6M'].notna().sum() >= len(df) * 0.5  # 절반 이상 데이터 있을 때만 반영

    # 가중치: 가치(80%) + 모멘텀(20%) — 데이터 없으면 가치 100%
    if has_mom:
        W = {'ROE': 0.24, 'PER': 0.24, 'PSR': 0.16, 'PBR': 0.16, 'MOM': 0.20}
    else:
        W = {'ROE': 0.30, 'PER': 0.30, 'PSR': 0.20, 'PBR': 0.20, 'MOM': 0.00}

    def _mark(val, metric):
        if metric == 'ROE':    return '🟢' if val >= 15   else '🔴'
        if metric == 'PER':    return '🟢' if val <= 15   else '🔴'
        if metric == 'PSR':    return '🟢' if val <= 3    else '🔴'
        if metric == 'PBR':    return '🟢' if val <= 1.5  else '🔴'
        if metric == 'MOM_6M': return '🟢' if val >= 10  else ('🟡' if val >= 0 else '🔴')
        return '─'

    def _print_top10(top_df, score_col):
        print("  (🟢 기준 통과·좋음  /  🟡 보통  /  🔴 기준 미달·나쁨)\n")
        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            mom_str = ''
            if has_mom and pd.notna(row.get('MOM_6M')):
                m6  = row['MOM_6M']
                m12 = row['MOM_12M'] if pd.notna(row.get('MOM_12M')) else None
                m12_s = f' / 12M {m12:+.1f}%' if m12 is not None else ''
                mom_str = f'   모멘텀: {_mark(m6,"MOM_6M")} 6M {m6:+.1f}%{m12_s}'
            print(f"  {rank:>2}위. [{row['종목코드']}] {row['기업명']}"
                  f"  →  종합점수: {row[score_col]:.1f} / 100점")
            print(f"       ROE: {_mark(row['ROE'],'ROE')} {row['ROE']}%"
                  f"   PER: {_mark(row['PER'],'PER')} {row['PER']}"
                  f"   PSR: {_mark(row['PSR'],'PSR')} {row['PSR']}"
                  f"   PBR: {_mark(row['PBR'],'PBR')} {row['PBR']}{mom_str}")
            print(f"       개별점수 → ROE:{row['ROE_점수']:.0f}점  "
                  f"PER:{row['PER_점수']:.0f}점  "
                  f"PSR:{row['PSR_점수']:.0f}점  "
                  f"PBR:{row['PBR_점수']:.0f}점"
                  + (f"  MOM:{row.get('MOM_점수',0):.0f}점" if has_mom else ''))
            print("  " + "-" * 70)

    # ════════════════════════════════════════════════════════════════════
    # 방법 A — 퍼센타일 가중 점수법
    # 적합 상황: 이 50개 중 상대적으로 가장 저렴한 종목을 찾을 때
    #            (군집 내 상대 위치 평가 → 항상 TOP10 도출됨)
    # ════════════════════════════════════════════════════════════════════
    df_a = df.copy()
    df_a['ROE_점수'] = df_a['ROE'].rank(pct=True) * 100
    df_a['PER_점수'] = (1 - df_a['PER'].rank(pct=True)) * 100
    df_a['PSR_점수'] = (1 - df_a['PSR'].rank(pct=True)) * 100
    df_a['PBR_점수'] = (1 - df_a['PBR'].rank(pct=True)) * 100
    if has_mom:
        mom_composite = df_a[['MOM_6M','MOM_12M']].mean(axis=1)
        df_a['MOM_점수'] = mom_composite.rank(pct=True) * 100
    else:
        df_a['MOM_점수'] = 0
    df_a['종합점수_A'] = (
        W['ROE'] * df_a['ROE_점수'] +
        W['PER'] * df_a['PER_점수'] +
        W['PSR'] * df_a['PSR_점수'] +
        W['PBR'] * df_a['PBR_점수'] +
        W['MOM'] * df_a['MOM_점수']
    )
    top10_a = df_a.sort_values('종합점수_A', ascending=False).head(10)

    mom_note = '가치80%+모멘텀20%' if has_mom else '가치100% (모멘텀 데이터 부족)'
    print("\n" + "=" * 70)
    print(f"📊 [방법 A] 퍼센타일 가중 점수법   TOP 10  |  섹터: {sector_name}")
    print(f"📌 가중치: {mom_note}  |  섹터 내 상대 위치 평가")
    print("=" * 70)
    _print_top10(top10_a, '종합점수_A')

    # ════════════════════════════════════════════════════════════════════
    # 방법 B — 절대 기준 구간 점수법
    # 적합 상황: 가치투자 절대 기준으로 통과하는 종목인지 평가할 때
    #            (기준 미달 시 감점 → 50개 전부 고평가면 모두 저점수 가능)
    # ════════════════════════════════════════════════════════════════════
    def _roe_score(v):
        if v >= 15: return 100
        if v >= 10: return 65
        if v >=  0: return 30
        return 0

    def _per_score(v):
        if v <= 15: return 100
        if v <= 25: return 65
        if v <= 40: return 30
        return 0

    def _psr_score(v):
        if v <= 3: return 100
        if v <= 5: return 60
        return 20

    def _pbr_score(v):
        if v <= 1.5: return 100
        if v <= 3.0: return 60
        return 20

    def _mom_score(v):
        if pd.isna(v): return 50  # 데이터 없으면 중립
        if v >= 30:  return 100
        if v >= 15:  return 80
        if v >= 0:   return 55
        if v >= -15: return 30
        return 10

    df_b = df.copy()
    df_b['ROE_점수'] = df_b['ROE'].apply(_roe_score)
    df_b['PER_점수'] = df_b['PER'].apply(_per_score)
    df_b['PSR_점수'] = df_b['PSR'].apply(_psr_score)
    df_b['PBR_점수'] = df_b['PBR'].apply(_pbr_score)
    if has_mom:
        mom_avg = df_b[['MOM_6M','MOM_12M']].mean(axis=1)
        df_b['MOM_점수'] = mom_avg.apply(_mom_score)
    else:
        df_b['MOM_점수'] = 0
    df_b['종합점수_B'] = (
        W['ROE'] * df_b['ROE_점수'] +
        W['PER'] * df_b['PER_점수'] +
        W['PSR'] * df_b['PSR_점수'] +
        W['PBR'] * df_b['PBR_점수'] +
        W['MOM'] * df_b['MOM_점수']
    )
    top10_b = df_b.sort_values('종합점수_B', ascending=False).head(10)

    print("\n" + "=" * 70)
    print(f"📊 [방법 B] 절대 기준 구간 점수법   TOP 10  |  섹터: {sector_name}")
    print(f"📌 가중치: {mom_note}  |  가치투자 절대 기준 평가")
    print("=" * 70)
    _print_top10(top10_b, '종합점수_B')