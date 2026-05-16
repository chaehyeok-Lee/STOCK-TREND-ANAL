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

        valid_stocks.append({
            '종목코드': ticker,
            '기업명':   info.get('shortName', ticker),
            'ROE':      round(roe, 2),
            'PER':      round(per, 2),
            'PSR':      round(psr, 2),
            'PBR':      round(pbr, 2),
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
    W = {'ROE': 0.30, 'PER': 0.30, 'PSR': 0.20, 'PBR': 0.20}

    def _mark(val, metric):
        if metric == 'ROE': return '🟢' if val >= 15  else '🔴'
        if metric == 'PER': return '🟢' if val <= 15  else '🔴'
        if metric == 'PSR': return '🟢' if val <= 3   else '🔴'
        if metric == 'PBR': return '🟢' if val <= 1.5 else '🔴'

    def _print_top10(top_df, score_col):
        print("  (🟢 기준 통과 · 좋음  /  🔴 기준 미달 · 나쁨)\n")
        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"  {rank:>2}위. [{row['종목코드']}] {row['기업명']}"
                  f"  →  종합점수: {row[score_col]:.1f} / 100점")
            print(f"       ROE: {_mark(row['ROE'],'ROE')} {row['ROE']}%"
                  f"   PER: {_mark(row['PER'],'PER')} {row['PER']}"
                  f"   PSR: {_mark(row['PSR'],'PSR')} {row['PSR']}"
                  f"   PBR: {_mark(row['PBR'],'PBR')} {row['PBR']}")
            print(f"       개별점수 → ROE:{row['ROE_점수']:.0f}점  "
                  f"PER:{row['PER_점수']:.0f}점  "
                  f"PSR:{row['PSR_점수']:.0f}점  "
                  f"PBR:{row['PBR_점수']:.0f}점")
            print("  " + "-" * 65)

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
    df_a['종합점수_A'] = (
        W['ROE'] * df_a['ROE_점수'] +
        W['PER'] * df_a['PER_점수'] +
        W['PSR'] * df_a['PSR_점수'] +
        W['PBR'] * df_a['PBR_점수']
    )
    top10_a = df_a.sort_values('종합점수_A', ascending=False).head(10)

    print("\n" + "=" * 70)
    print(f"📊 [방법 A] 퍼센타일 가중 점수법   TOP 10  |  섹터: {sector_name}")
    print("📌 적합 상황: 이 섹터 안에서 상대적으로 가장 저렴한 종목을 찾을 때")
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

    df_b = df.copy()
    df_b['ROE_점수'] = df_b['ROE'].apply(_roe_score)
    df_b['PER_점수'] = df_b['PER'].apply(_per_score)
    df_b['PSR_점수'] = df_b['PSR'].apply(_psr_score)
    df_b['PBR_점수'] = df_b['PBR'].apply(_pbr_score)
    df_b['종합점수_B'] = (
        W['ROE'] * df_b['ROE_점수'] +
        W['PER'] * df_b['PER_점수'] +
        W['PSR'] * df_b['PSR_점수'] +
        W['PBR'] * df_b['PBR_점수']
    )
    top10_b = df_b.sort_values('종합점수_B', ascending=False).head(10)

    print("\n" + "=" * 70)
    print(f"📊 [방법 B] 절대 기준 구간 점수법   TOP 10  |  섹터: {sector_name}")
    print("📌 적합 상황: 가치투자 절대 기준으로 통과하는 종목인지 평가할 때")
    print("=" * 70)
    _print_top10(top10_b, '종합점수_B')