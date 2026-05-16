import yfinance as yf
import pandas as pd

SEP    = '=' * 65
DASH   = '─' * 65
BORDER = '█' * 65
REGIME_KR = {'bullish': '강세장 🟢', 'neutral': '중립장 🟡', 'bearish': '약세장 🔴'}

WATCHLIST = {
    '반도체': ['NVDA', 'AMD', 'AVGO', 'TSM', 'AMAT', 'KLAC', 'LRCX', 'MU', 'QCOM', 'ASML'],
    '소프트웨어': ['MSFT', 'GOOGL', 'META', 'CRM', 'NOW', 'ADBE', 'PLTR', 'SNOW', 'ORCL'],
    '헬스케어': ['LLY', 'NVO', 'ABBV', 'UNH', 'ISRG', 'DXCM', 'MRNA', 'GILD', 'VRTX'],
}

HOLDING_PARAMS = {
    '1m': {
        'label': '단기 스윙 (1개월)', 'atr_mult': 1.5, 'trail_mult': 1.0,
        'tp_pcts': [0.08, 0.15], 'split_ratios': [0.50, 0.50],
        'entry_drops': [0.0, -0.04], 'monitor': '매일 종가 확인',
        'rebal_months': 1, 'note': '모멘텀 강할 때 단기 차익. 빠른 손절 필수.',
    },
    '3m': {
        'label': '중기 포지션 (3개월)', 'atr_mult': 2.0, 'trail_mult': 1.5,
        'tp_pcts': [0.10, 0.20], 'split_ratios': [0.33, 0.33, 0.34],
        'entry_drops': [0.0, -0.05, -0.10], 'monitor': '주 1회 확인',
        'rebal_months': 3, 'note': '분기 실적 사이클 활용. 분할 매수로 리스크 분산.',
    },
    '6m': {
        'label': '장기 성장 (6개월)', 'atr_mult': 3.0, 'trail_mult': 2.0,
        'tp_pcts': [0.20, 0.40], 'split_ratios': [0.25, 0.25, 0.25, 0.25],
        'entry_drops': [0.0, -0.05, -0.10, -0.15], 'monitor': '월 1회 확인',
        'rebal_months': 6, 'note': '성장 스토리 베팅. 단기 변동성 무시, 넓은 손절.',
    },
}


# ── 기술적 계산 ───────────────────────────────────────────────────────────────

def _atr(hist, period=14):
    if hist.empty or len(hist) < period + 1:
        return None
    if not all(c in hist.columns for c in ['High', 'Low', 'Close']):
        return None
    hl  = hist['High'] - hist['Low']
    hpc = (hist['High'] - hist['Close'].shift(1)).abs()
    lpc = (hist['Low']  - hist['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return round(float(val), 2) if pd.notna(val) else None


def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float('nan'))
    val      = (100 - 100 / (1 + rs)).fillna(50).iloc[-1]
    return round(float(val), 1)


def _macd_signal(close):
    ef = close.ewm(span=12, adjust=False).mean()
    es = close.ewm(span=26, adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=9, adjust=False).mean()
    return float(m.iloc[-1]) > float(s.iloc[-1])


def _ann_vol(close):
    ret = close.pct_change().dropna()
    if len(ret) < 20:
        return None
    return round(float(ret.std() * (252 ** 0.5) * 100), 1)


def _bar(pct, width=20):
    filled = max(0, min(width, int(round(pct / 100 * width))))
    return '█' * filled + '░' * (width - filled)


# ── 시장 환경 / 실적 / 켈리 / 판정 ──────────────────────────────────────────

def _get_market_context():
    ctx = {'regime': 'neutral', 'spy_price': None, 'ma200': None,
           'vix': None, 'spy_ret_20d': None}
    try:
        h = yf.Ticker('SPY').history(period='1y')
        if not h.empty:
            if h.index.tz is not None:
                h.index = h.index.tz_convert(None)
            c = h['Close']
            ctx['spy_price'] = round(float(c.iloc[-1]), 2)
            if len(c) >= 200:
                m = round(float(c.rolling(200).mean().iloc[-1]), 2)
                ctx['ma200'] = m
                sp = ctx['spy_price']
                ctx['regime'] = ('bullish' if sp > m
                                 else 'bearish' if sp < m * 0.95 else 'neutral')
            if len(c) >= 20:
                ctx['spy_ret_20d'] = round(
                    (float(c.iloc[-1]) / float(c.iloc[-20]) - 1) * 100, 1)
    except Exception:
        pass
    try:
        vh = yf.Ticker('^VIX').history(period='5d')
        if not vh.empty:
            ctx['vix'] = round(float(vh['Close'].iloc[-1]), 1)
    except Exception:
        pass
    return ctx


def _get_earnings_info(stock):
    try:
        cal = stock.calendar
        if not cal:
            try:
                ed_dates = stock.earnings_dates
                if ed_dates is not None and not ed_dates.empty:
                    now2 = pd.Timestamp.now()
                    idx  = ed_dates.index
                    if hasattr(idx, 'tz') and idx.tz is not None:
                        idx = idx.tz_convert(None)
                    future = sorted(d for d in idx if d >= now2)
                    if future:
                        ed = future[0]
                        return ed, int((ed - now2).days)
            except Exception:
                pass
            return None, None

        now = pd.Timestamp.now()
        ed  = None

        if isinstance(cal, dict):
            val = cal.get('Earnings Date') or cal.get('earningsDate')
            if val is not None:
                dates = ([val] if (not hasattr(val, '__iter__') or isinstance(val, str))
                         else list(val))
                parsed = []
                for d in dates:
                    try:
                        ts = pd.Timestamp(d)
                        if ts.tzinfo:
                            ts = ts.tz_convert(None)
                        parsed.append(ts)
                    except Exception:
                        pass
                future = sorted(d for d in parsed if d >= now)
                ed = future[0] if future else (max(parsed) if parsed else None)

        elif hasattr(cal, 'index'):
            for key in ('Earnings Date', 'Earnings date'):
                if key in cal.index:
                    try:
                        raw = cal.loc[key]
                        val = raw.iloc[0] if hasattr(raw, 'iloc') else raw
                        ts  = pd.Timestamp(val)
                        if ts.tzinfo:
                            ts = ts.tz_convert(None)
                        ed = ts
                        break
                    except Exception:
                        pass

        if ed is None:
            return None, None
        return ed, int((ed - now).days)
    except Exception:
        return None, None


def _dynamic_kelly_wr(regime, vix_val):
    wr = 0.55
    if regime == 'bullish':
        wr += 0.05
    elif regime == 'bearish':
        wr -= 0.05
    if vix_val is not None:
        if vix_val >= 30:
            wr -= 0.05
        elif vix_val >= 20:
            wr -= 0.02
    return round(max(0.40, min(0.70, wr)), 2)


def _go_no_go(rsi_val, macd_bull, rr, regime, vix_val,
              days_until_earnings, pos52, hk_pct):
    signals, score = [], 0

    if rsi_val is not None:
        if rsi_val > 70:
            signals.append(('RSI 과매수', -1, f'{rsi_val}'))
            score -= 1
        elif rsi_val < 30:
            signals.append(('RSI 과매도(반등)', +1, f'{rsi_val}'))
            score += 1
        else:
            signals.append(('RSI 중립', 0, f'{rsi_val}'))

    if macd_bull is not None:
        if macd_bull:
            signals.append(('MACD 상승', +1, '단기>장기 이평'))
            score += 1
        else:
            signals.append(('MACD 하락', -1, '단기<장기 이평'))
            score -= 1

    if rr is not None:
        if rr >= 2.0:
            signals.append(('R/R 우수', +1, f'{rr}:1'))
            score += 1
        elif rr >= 1.5:
            signals.append(('R/R 보통', 0, f'{rr}:1'))
        else:
            signals.append(('R/R 불량', -1, f'{rr}:1'))
            score -= 1

    if regime == 'bullish':
        signals.append(('시장 강세', +1, 'SPY>200MA'))
        score += 1
    elif regime == 'bearish':
        signals.append(('시장 약세', -1, 'SPY<200MA×0.95'))
        score -= 1
    else:
        signals.append(('시장 중립', 0, 'SPY≈200MA'))

    if vix_val is not None:
        if vix_val >= 30:
            signals.append(('VIX 공포구간', -1, f'{vix_val}'))
            score -= 1
        elif vix_val >= 20:
            signals.append(('VIX 주의구간', 0, f'{vix_val}'))
        else:
            signals.append(('VIX 안정구간', +1, f'{vix_val}'))
            score += 1

    if days_until_earnings is not None and 0 <= days_until_earnings <= 14:
        signals.append(('실적 임박', -1, f'{days_until_earnings}일 후'))
        score -= 1

    if pos52 is not None:
        if pos52 > 75:
            signals.append(('52주 고점권', -1, f'{pos52}%'))
            score -= 1
        elif pos52 < 25:
            signals.append(('52주 저점권', +1, f'{pos52}%'))
            score += 1

    if hk_pct is not None:
        if hk_pct > 0:
            signals.append(('켈리 양수', +1, f'{hk_pct}%'))
            score += 1
        else:
            signals.append(('켈리 음수/제로', -1, '기대값≤0'))
            score -= 1

    if score >= 3:
        verdict, emoji = 'GO — 매수 추천', '🟢'
    elif score <= -2:
        verdict, emoji = 'NO-GO — 비추천', '🔴'
    else:
        verdict, emoji = 'HOLD — 보류 (추가 확인)', '🟡'

    return signals, score, verdict, emoji


# ── 방법론 설명 ───────────────────────────────────────────────────────────────

def show_methodology():
    print(f"\n{SEP}")
    print(f"  📚  투자 방법론 — 실제 사례 & 근거")
    print(SEP)

    sections = [
        ("1. ATR 손절 (Average True Range Stop-Loss)",
         ["실사례: 터틀 트레이더 (Richard Dennis, 1983)",
          "  → ATR 기반 추세추종으로 연평균 약 18% 수익",
          "  → Man AHL, Winton Group 등 CTA 헤지펀드가 변형 사용",
          "원리  : 최근 변동폭(ATR) 배수를 손절 기준으로 삼아",
          "       시장 노이즈 안에서는 버티고 진짜 추세 전환 시 청산",
          "리스크: 횡보장에서 잦은 손절 → 수수료·슬리피지 누적"]),
        ("2. 하프켈리 포지션 사이징 (Half-Kelly Criterion)",
         ["실사례: Ed Thorp (블랙잭 → 헤지펀드, 1960~2000년대)",
          "  → Kelly 공식으로 수십억 달러 운용, 230개월 무손실",
          "  → 르네상스 테크놀로지, DE Shaw 등 퀀트 펀드 적용",
          "원리  : 수학적 최적 베팅의 절반만 사용 → 파산 리스크 감소",
          "       f = (승률×수익 − 패율×손실) ÷ 수익",
          "★ 개선: 고정 0.55 → 시장국면+VIX 기반 동적 계산",
          "       강세+VIX안정 → 0.60 / 약세+VIX공포 → 0.45",
          "리스크: 승률 추정 오류 시 과도 베팅 → 보수적 가정 권장"]),
        ("3. 분할매수 DCA (Dollar Cost Averaging)",
         ["실사례: 워런 버핏의 점진적 포지션 구축 방식",
          "  → 기관투자자 표준 (VWAP 기반 분할 체결)",
          "  → S&P500 장기 DCA: 과거 20년 기준 연평균 약 10%",
          "원리  : 진입 시점 분산 → 평균 단가 낮추기 + 타이밍 리스크 축소",
          "★ 개선: SPY 대비 상대 수익률로 시장 하락 vs 종목 문제 구분",
          "리스크: 하락 추세 지속 시 평균단가 상승 → 손실 확대 가능"]),
    ]

    for title, lines in sections:
        print(f"\n{DASH}")
        print(f"  {title}")
        print(DASH)
        for line in lines:
            print(f"  {line}")

    print(f"\n{DASH}")
    print(f"  ※ 세 방법 조합 (이 툴의 접근법)")
    print(DASH)
    print(f"  ATR 손절 + 하프켈리 비중 + DCA 진입의 조합은")
    print(f"  체계적 트레이더들이 실전에서 가장 많이 사용하는 리스크 관리 체계입니다.")
    print(f"  과거 성과가 미래를 보장하지 않으며 개인 판단이 최우선입니다.")
    print(f"\n{SEP}\n")


# ── 용어 사전 ─────────────────────────────────────────────────────────────────

def show_glossary():
    print(f"\n{SEP}")
    print(f"  📖  투자 용어 사전  (복잡한 지표를 쉽게 이해하기)")
    print(SEP)

    terms = [
        ("ATR (Average True Range) — 평균 진폭",
         "최근 14일간 하루 평균 얼마나 오르내렸는지를 나타내는 숫자",
         "ATR=$5 → 하루 평균 $5 움직임. 이 값의 2배를 손절 폭으로 씁니다"),
        ("RSI (Relative Strength Index) — 상대 강도 지수",
         "0~100 사이 숫자. 최근 얼마나 빠르게, 강하게 올랐는지 측정",
         "70↑ = 과열(너무 빨리 올라 조정 가능) / 30↓ = 침체(반등 기대) / 40~60 = 정상"),
        ("MACD — 이동평균 수렴·발산",
         "12일 평균과 26일 평균의 차이. 단기 추세와 장기 추세를 비교",
         "단기선이 장기선을 위로 돌파 = 상승 신호(골든크로스) / 아래로 = 하락(데드크로스)"),
        ("200MA (200일 이동평균선)",
         "최근 200거래일(약 10개월) 평균 주가. 장기 추세의 나침반",
         "주가 > 200MA → 강세. SPY가 200MA 위 → 전체 시장 강세장으로 판단"),
        ("VIX — 공포 지수",
         "S&P500 옵션 시장이 예상하는 향후 30일 변동성. 투자자의 불안감 온도계",
         "20↓ = 평온 / 20~30 = 주의 / 30↑ = 공포(급락 위험, 포지션 50% 자동 축소)"),
        ("R/R (Risk/Reward Ratio) — 위험 대비 보상",
         "얼마 잃을 수 있는가(위험) vs 얼마 벌 수 있는가(보상)의 비율",
         "R/R=2:1 → 목표수익 +20% / 손절 -10%. '2번 틀려도 1번에 본전' 구조. 최소 1.5:1 권장"),
        ("하프켈리 (Half-Kelly Criterion)",
         "수학적으로 최적인 베팅 비율의 절반. 과도한 리스크를 막는 안전장치",
         "100% 켈리는 수익은 크지만 변동폭도 큼. 절반만 쓰면 안정적 복리 성장"),
        ("P/E (Price-to-Earnings) — 주가수익비율",
         "현재 주가가 연간 이익의 몇 배인지. '이 회사 이익의 몇 년치를 지금 내가 사는가'",
         "P/E=25 → 25년치 이익 지불. 낮을수록 저평가. 반드시 같은 섹터 평균과 비교"),
        ("PEG (Price/Earnings-to-Growth) — 성장률 감안 P/E",
         "P/E를 성장률로 나눈 값. 빠르게 성장하는 기업의 비싼 P/E를 정당화",
         "PEG < 1 = 성장 대비 저평가 / PEG = 1 = 적정 / PEG > 2 = 고평가"),
        ("FCF Yield (Free Cash Flow Yield) — 잉여현금흐름 수익률",
         "회사가 실제로 벌어들이는 현금을 시가총액으로 나눈 비율. '진짜 수익률'",
         "FCF Yield=5% → 시총 대비 5% 현금 창출. 높을수록 배당·자사주·투자 여력 큼"),
        ("Beta — 시장 민감도",
         "시장(SPY)이 1% 움직일 때 이 주식이 몇 % 움직이는지",
         "Beta=1.5 → 시장+10%면 +15%, 시장-10%면 -15%. 1보다 크면 공격적, 작으면 방어적"),
        ("D/E (Debt-to-Equity) — 부채비율",
         "자본 대비 빚이 얼마나 많은지. 금리 인상 시 영향을 받는 정도",
         "D/E=100% → 빚=자본. 200% 이상은 금리 민감. 기술주는 낮고 유틸리티는 높은 경향"),
        ("공매도 비율 (Short % of Float)",
         "유통 주식 중 하락에 베팅한(공매도) 비율. 시장의 부정적 견해 온도계",
         "5% 미만=정상 / 10% 이상=위험(숏스퀴즈 or 펀더멘털 문제 신호)"),
        ("VWAP (Volume Weighted Average Price)",
         "거래량을 고려한 평균 가격. 기관투자자가 대량 매수/매도 시 기준으로 삼는 가격",
         "VWAP 근처에서 분할 체결하면 시장 충격(슬리피지) 최소화 가능"),
        ("슬리피지 (Slippage)",
         "원하는 가격과 실제 체결된 가격의 차이. 대량 매수 시 가격이 밀려 발생",
         "1000주를 $100에 사려 했는데 $100.5에 체결 → 슬리피지 $0.5/주"),
        ("배당수익률 (Dividend Yield)",
         "주가 대비 연간 배당금 비율. 주식을 보유하는 것만으로 받는 '이자' 같은 개념",
         "주가 $100, 연 배당금 $3 → 배당수익률 3%. 성장주(NVDA)는 낮고 배당주(JNJ)는 높음"),
        ("손실 회복 비대칭 (Recovery Asymmetry)",
         "손실 후 원금 회복에 필요한 수익률이 항상 손실률보다 크다는 수학적 사실",
         "−10% 손실 → +11.1% 필요 / −20% 손실 → +25% 필요 / −50% 손실 → +100% 필요"),
    ]

    for name, desc, example in terms:
        print(f"\n{DASH}")
        print(f"  ▶ {name}")
        print(f"  {desc}")
        print(f"  예시: {example}")

    print(f"\n{SEP}\n")


# ── 단일 종목 분석 ────────────────────────────────────────────────────────────

def analyze_portfolio(ticker, capital, holding='3m'):
    params = HOLDING_PARAMS.get(holding, HOLDING_PARAMS['3m'])

    print(f"\n{SEP}")
    print(f"  💼  [{ticker}]  매매 플랜  |  투자금: ${capital:,.0f}  |  {params['label']}")
    print(SEP)
    print("  시장 환경 + 종목 데이터 수집 중...")

    mkt         = _get_market_context()
    regime      = mkt['regime']
    vix_val     = mkt['vix']
    spy_ret_20d = mkt['spy_ret_20d']

    vix_size_mult     = 0.5 if (vix_val is not None and vix_val >= 30) else 1.0
    effective_capital = capital * vix_size_mult

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        hist  = stock.history(period='1y')
    except Exception as e:
        print(f"  ❌ [{ticker}] 데이터 수집 실패: {type(e).__name__}")
        print(f"     네트워크 연결 또는 티커를 확인해주세요.")
        print(f"{SEP}\n")
        return

    if hist.empty or not info:
        print(f"  ❌ [{ticker}] 데이터 없음 — 티커가 올바른지 확인해주세요.")
        print(f"{SEP}\n")
        return
    if len(hist) < 30:
        print(f"  ⚠️  [{ticker}] 데이터 부족 ({len(hist)}일). 최소 30일 필요.")
        print(f"{SEP}\n")
        return

    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)

    earnings_date, days_until_earnings = _get_earnings_info(stock)

    name      = info.get('longName') or info.get('shortName') or ticker
    sector    = info.get('sector', 'N/A')
    industry  = info.get('industry', 'N/A')
    close     = hist['Close']
    price     = round(float(close.iloc[-1]), 2)
    h52       = info.get('fiftyTwoWeekHigh')
    l52       = info.get('fiftyTwoWeekLow')
    target    = info.get('targetMeanPrice')
    t_high    = info.get('targetHighPrice')
    t_low     = info.get('targetLowPrice')
    n_anal    = int(info.get('numberOfAnalystOpinions') or 0)
    rec       = info.get('recommendationKey', '')
    short_pct = info.get('shortPercentOfFloat')

    # 펀더멘털
    pe_trail  = info.get('trailingPE')
    pe_fwd    = info.get('forwardPE')
    peg       = info.get('trailingPegRatio') or info.get('pegRatio')
    beta      = info.get('beta')
    de_ratio  = info.get('debtToEquity')
    _div_raw  = info.get('dividendYield')
    div_yield = round(_div_raw, 2) if (_div_raw and 0 < _div_raw < 25) else None
    fcf       = info.get('freeCashflow')
    mktcap    = info.get('marketCap')
    fcf_yield = (round(fcf / mktcap * 100, 1)
                 if (fcf and mktcap and mktcap > 0) else None)

    avg_vol_shares = (float(hist['Volume'].tail(20).mean())
                      if 'Volume' in hist.columns else None)
    avg_dollar_vol = round(avg_vol_shares * price) if avg_vol_shares else None

    atr_val       = _atr(hist)
    rsi_val       = _rsi(close)
    vol_val       = _ann_vol(close)
    macd_bull     = _macd_signal(close)
    stock_ret_20d = (round((float(close.iloc[-1]) / float(close.iloc[-20]) - 1) * 100, 1)
                     if len(close) >= 20 else None)

    pos52      = (round((price - l52) / (h52 - l52) * 100, 1)
                  if (h52 and l52 and h52 != l52) else None)
    upside_pct = round((target - price) / price * 100, 1) if target else None

    atr_mult     = params['atr_mult']
    trail_mult   = params['trail_mult']
    tp_pcts      = params['tp_pcts']
    split_ratios = params['split_ratios']
    entry_drops  = params['entry_drops']

    stop_loss  = round(price - atr_mult * atr_val, 2)  if atr_val else None
    risk_pct   = round(atr_mult * atr_val / price * 100, 1) if atr_val else None
    trail_stop = round(price - trail_mult * atr_val, 2) if atr_val else None

    entries      = [round(price * (1 + d), 2) for d in entry_drops]
    amounts      = [effective_capital * r for r in split_ratios]
    shares_per   = [max(1, int(a // e)) for a, e in zip(amounts, entries)]
    total_shares = sum(shares_per)
    avg_cost     = (round(sum(s * e for s, e in zip(shares_per, entries)) / total_shares, 2)
                    if total_shares else price)

    tps          = [round(price * (1 + p), 2) for p in tp_pcts]
    tp_auto_last = round(price * (1 + tp_pcts[-1] * 1.5), 2)
    if target and round(target, 2) > tps[-1]:
        tp_final = round(target, 2)
        tp_label = '월가 컨센서스'
    else:
        tp_final = tp_auto_last
        tp_label = '자체 계산 (월가 목표가 낮음)'

    sh_imm   = max(1, int(effective_capital // price))
    cost_imm = round(sh_imm * price, 2)
    ml_imm   = round(sh_imm * atr_mult * atr_val, 2) if atr_val else None
    ml_split = round(total_shares * atr_mult * atr_val, 2) if atr_val else None

    if upside_pct and risk_pct and risk_pct > 0:
        rr       = round(upside_pct / risk_pct, 2)
        rr_label = '우수 ✅' if rr >= 2.0 else '보통 🟡' if rr >= 1.5 else '주의 🔴'
    else:
        rr = rr_label = None

    w = _dynamic_kelly_wr(regime, vix_val)
    if upside_pct and risk_pct and upside_pct > 0 and risk_pct > 0:
        kelly_f = (w * (upside_pct / 100) - (1 - w) * (risk_pct / 100)) / (upside_pct / 100)
        hk_f    = max(0.0, min(kelly_f / 2, 1.0))
        hk_pct  = round(hk_f * 100, 1)
        hk_amt  = round(effective_capital * hk_f, 0)
    else:
        hk_pct = hk_amt = None

    rebal = (pd.Timestamp.now() + pd.DateOffset(months=params['rebal_months'])).strftime('%Y-%m-%d')

    # ── 출력 시작 ──────────────────────────────────────────────────
    print(f"\n  🏢  {name}")
    print(f"      섹터: {sector}  |  산업: {industry}")
    print(f"      애널리스트 {n_anal}명  |  컨센서스: {rec or 'N/A'}")
    if short_pct is not None:
        sc = '🔴' if short_pct > 0.10 else '🟡' if short_pct > 0.05 else '🟢'
        print(f"      공매도 비율: {sc} {short_pct * 100:.1f}%  (10%↑위험, 5%↑주의)")
    print(f"      전략 메모: {params['note']}")

    # 시장 환경
    print(f"\n{DASH}")
    print(f"  🌍 시장 환경  (전략 자동 반영)")
    print(f"{DASH}")
    if mkt['spy_price'] and mkt['ma200']:
        spy_vs_ma = round((mkt['spy_price'] / mkt['ma200'] - 1) * 100, 1)
        print(f"  SPY / 200MA  : ${mkt['spy_price']} / ${mkt['ma200']}  ({spy_vs_ma:+.1f}%)")
        print(f"                 ※ 200MA=10개월 평균선. 위=강세, 아래=약세")
    print(f"  시장 국면    : {REGIME_KR[regime]}")
    if vix_val is not None:
        vix_lbl = ('공포구간 🔴 → 포지션 50% 자동 축소' if vix_val >= 30
                   else '주의구간 🟡' if vix_val >= 20 else '안정구간 🟢')
        print(f"  VIX (공포지수): {vix_val}  →  {vix_lbl}")
        print(f"                 ※ 20↑주의, 30↑공포 (시장의 불안감 온도계)")
        if vix_val >= 30:
            print(f"  ⚠️  유효 투자금: ${effective_capital:,.0f}  (원래 ${capital:,.0f}의 50%)")
    print(f"  하프켈리 승률: {w:.2f}  (국면+VIX 기반 동적 계산)")

    # 실적 발표 경고
    if earnings_date and days_until_earnings is not None and days_until_earnings >= 0:
        if days_until_earnings <= 14:
            print(f"\n  ⚠️  【실적 발표 임박】 {earnings_date.strftime('%Y-%m-%d')} ({days_until_earnings}일 후)")
            print(f"     실적 전후 급등락 위험 — 포지션 축소 or 발표 후 진입 권장")
        elif days_until_earnings <= 45:
            print(f"  📅 다음 실적: {earnings_date.strftime('%Y-%m-%d')} ({days_until_earnings}일 후)")

    # 현재 상태
    print(f"\n{DASH}")
    print(f"  📍 현재 상태  (기술적 지표)")
    print(f"{DASH}")
    print(f"  현재가       : ${price}")
    if h52 and l52 and pos52 is not None:
        bar = _bar(pos52)
        print(f"  52주 범위    : ${l52:.2f} [{bar}] ${h52:.2f}")
        pos52_lbl = ('고점권 ⚠️ 추가 상승 여력 제한' if pos52 > 75
                     else '저점권 💡 반등 가능' if pos52 < 25 else '중간')
        print(f"  52주 위치    : {pos52}%  →  {pos52_lbl}")
    rsi_lbl = ('과매수 ⚠️ 단기 조정 가능' if rsi_val > 70
               else '과매도 💡 반등 기대' if rsi_val < 30 else '중립 (정상)')
    print(f"  RSI(14)      : {rsi_val}  →  {rsi_lbl}")
    print(f"  ATR(14)      : ${atr_val}  ← 오늘 예상 변동폭, 손절 계산 기준")
    print(f"  연간 변동성  : {vol_val}%")
    macd_lbl = ('상승 모멘텀 ▲ (단기 평균이 장기 평균 위)' if macd_bull
                else '하락 모멘텀 ▼ (단기 평균이 장기 평균 아래)')
    print(f"  MACD         : {macd_lbl}")
    if target:
        range_str = f"  범위 ${t_low:.2f}~${t_high:.2f}" if (t_low and t_high) else ""
        print(f"  월가 목표가  : ${target:.2f}  ({upside_pct:+.1f}%){('  |  ' + range_str) if range_str else ''}")

    # 펀더멘털 지표
    print(f"\n{DASH}")
    print(f"  💹 펀더멘털 지표  (기업 가치 평가)")
    print(f"{DASH}")
    has_fund = False
    if pe_trail is not None and pe_trail > 0:
        print(f"  P/E (현재)   : {pe_trail:.1f}배  ← 현재 이익 기준. 낮을수록 저평가")
        has_fund = True
    if pe_fwd is not None and pe_fwd > 0:
        print(f"  P/E (예상)   : {pe_fwd:.1f}배  ← 내년 이익 기준. 현재보다 낮으면 성장 기대")
        has_fund = True
    if peg is not None and peg > 0:
        peg_lbl = '저평가 💡' if peg < 1.0 else '적정' if peg <= 1.5 else '주의 ⚠️'
        print(f"  PEG 비율     : {peg:.2f}  →  {peg_lbl}  (1.0↓=성장 대비 저평가)")
        has_fund = True
    if beta is not None:
        beta_lbl = ('저변동 방어주' if beta < 0.8 else
                    '시장 동조' if beta <= 1.2 else '고변동 공격주')
        print(f"  Beta         : {beta:.2f}  →  {beta_lbl}")
        print(f"                 ※ 시장+10%면 이 종목 {beta*10:+.1f}% / 시장-10%면 {-beta*10:+.1f}%")
        has_fund = True
    if de_ratio is not None:
        de_lbl = ('부채 낮음 💡' if de_ratio < 50 else
                  '적정' if de_ratio < 150 else '부채 높음 ⚠️')
        print(f"  부채비율(D/E): {de_ratio:.0f}%  →  {de_lbl}  (자본 대비 빚)")
        has_fund = True
    if fcf_yield is not None:
        fy_lbl = '우수 💡' if fcf_yield > 5 else '양호' if fcf_yield > 2 else '낮음'
        print(f"  FCF Yield    : {fcf_yield:.1f}%  →  {fy_lbl}  (실현 현금 수익률)")
        has_fund = True
    if div_yield is not None and div_yield > 0:
        print(f"  배당수익률   : {div_yield:.2f}%  ← 주가 대비 연간 배당금 비율")
        has_fund = True
    if not has_fund:
        print(f"  (펀더멘털 데이터 없음 — ETF 또는 데이터 미제공)")

    # 보유기간 비교
    print(f"\n{DASH}")
    print(f"  📅 보유기간별 비교  (선택: {params['label']})")
    print(f"{DASH}")
    if atr_val:
        print(f"  {'구분':<14} {'단기(1개월)':>13} {'중기(3개월)':>13} {'장기(6개월)':>13}")
        print(f"  {'─'*13} {'─'*13} {'─'*13} {'─'*13}")
        stops  = {k: round(price - HOLDING_PARAMS[k]['atr_mult'] * atr_val, 2) for k in ['1m', '3m', '6m']}
        tp1s   = {k: round(price * (1 + HOLDING_PARAMS[k]['tp_pcts'][0]), 2) for k in ['1m', '3m', '6m']}
        tp2s   = {k: round(price * (1 + HOLDING_PARAMS[k]['tp_pcts'][1]), 2) for k in ['1m', '3m', '6m']}
        splits = {'1m': '2회(50/50)', '3m': '3회(33/34)', '6m': '4회(25×4)'}
        print(f"  {'손절가':<14} ${stops['1m']:>11.2f}  ${stops['3m']:>11.2f}  ${stops['6m']:>11.2f}")
        print(f"  {'1차 익절':<14} ${tp1s['1m']:>11.2f}  ${tp1s['3m']:>11.2f}  ${tp1s['6m']:>11.2f}")
        print(f"  {'2차 익절':<14} ${tp2s['1m']:>11.2f}  ${tp2s['3m']:>11.2f}  ${tp2s['6m']:>11.2f}")
        print(f"  {'분할 방식':<14} {splits['1m']:>13} {splits['3m']:>13} {splits['6m']:>13}")

    # 리스크/보상
    print(f"\n{DASH}")
    print(f"  📊 리스크 / 보상 분석")
    print(f"{DASH}")
    if rr:
        print(f"  R/R 비율     : {rr}:1  →  {rr_label}")
        print(f"  기대 수익    : +{upside_pct}%  (월가 목표가 기준)")
        print(f"  최대 손실    : −{risk_pct}%  (ATR×{atr_mult} 손절)")
        print(f"                 ※ R/R≥2:1 → '2번 틀려도 1번에 본전' 구조")
    if hk_pct is not None:
        k_lbl = ('적정' if 0 < hk_pct <= 25 else
                 '과도 — 축소 권장' if hk_pct > 25 else '매수 비추천 (기대값 음수)')
        print(f"  동적 하프켈리: {hk_pct}%  (${hk_amt:,.0f})  →  {k_lbl}")
        print(f"  ※ 승률 {w:.2f} 적용  (국면:{regime} VIX:{vix_val})")

    # 진입 전략 1
    print(f"\n{DASH}")
    print(f"  📥 전략 1 — 즉시 전액 매수")
    if vix_size_mult < 1.0:
        print(f"  ⚠️  VIX≥30: 유효 투자금 ${effective_capital:,.0f}")
    print(f"{DASH}")
    print(f"  매수가    : ${entries[0]}  →  {sh_imm}주  (실투자금 ${cost_imm:,.2f})")
    if stop_loss:
        print(f"  손절가    : ${stop_loss}  (−{risk_pct}%,  ATR×{atr_mult})")
    if ml_imm:
        print(f"  최대손실  : −${ml_imm:,.2f}  (−{round(ml_imm / effective_capital * 100, 1)}%)")
    print(f"  적합 상황 : 상승 모멘텀 강하고 즉시 진입 확신 있을 때")

    # 진입 전략 2
    print(f"\n{DASH}")
    print(f"  📥 전략 2 — {len(entries)}회 분할매수  (권장)")
    if vix_size_mult < 1.0:
        print(f"  ⚠️  VIX≥30: 유효 투자금 ${effective_capital:,.0f}")
    print(f"{DASH}")
    labels = ['즉시' if d == 0.0 else f'−{abs(int(round(d * 100)))}% 하락' for d in entry_drops]
    for i, (e, sh, amt, ratio) in enumerate(zip(entries, shares_per, amounts, split_ratios)):
        print(f"  {i+1}차 ({labels[i]}, {int(ratio*100)}%)  : ${e}  →  {sh}주  (${amt:,.0f})")
    print(f"  ─ 평균단가: ${avg_cost}  |  총 {total_shares}주")
    if stop_loss:
        print(f"  ─ 손절가 : ${stop_loss}  (ATR×{atr_mult})")
    if ml_split:
        print(f"  ─ 최대손실: −${ml_split:,.2f}  (−{round(ml_split / effective_capital * 100, 1)}%)")
    print(f"  적합 상황 : 변동성 높거나 확신 부족할 때")

    # SPY 상대 수익률
    if stock_ret_20d is not None and spy_ret_20d is not None:
        rel = round(stock_ret_20d - spy_ret_20d, 1)
        print(f"\n  📊 SPY 상대 성과  (최근 20일 — 분할매수 추가 여부 판단)")
        print(f"  {ticker:<6} {stock_ret_20d:+.1f}%  /  SPY {spy_ret_20d:+.1f}%  /  초과: {rel:+.1f}%")
        if stock_ret_20d < 0 and spy_ret_20d < 0 and abs(rel) < 3:
            print(f"  → 시장 전반 하락 동조 — 분할매수 유효 (시장 반등 시 함께 회복)")
        elif rel < -5:
            print(f"  → ⚠️  개별 종목 문제 ({rel:.1f}% 언더퍼폼) — 원인 확인 후 추가 매수")
        elif rel > 5:
            print(f"  → 📈 시장 대비 강세 — 1차 진입 우선 고려")
        else:
            print(f"  → 시장과 유사 — 분할매수 무방")

    # 유동성 체크
    if avg_dollar_vol and avg_dollar_vol > 0:
        liq_thr = avg_dollar_vol * 0.01
        print(f"\n  💧 유동성 체크  (일평균 거래대금: ${avg_dollar_vol:,.0f})")
        if effective_capital > liq_thr:
            print(f"  ⚠️  포지션 ${effective_capital:,.0f} > 거래량 1% ${liq_thr:,.0f}")
            print(f"     VWAP 분할 체결 또는 포지션 축소 권장")
        else:
            print(f"  ✅ 유동성 양호  ({effective_capital / avg_dollar_vol * 100:.2f}%)")

    # 익절 전략
    tp_sell_pcts = [30, 30, 40] if len(tps) >= 2 else [50, 50]
    print(f"\n{DASH}")
    print(f"  🎯 익절 전략")
    print(f"{DASH}")
    for i, (tp, sp) in enumerate(zip(tps, tp_sell_pcts)):
        print(f"  {i+1}차 (+{int(tp_pcts[i]*100)}%)  : ${tp}  →  보유의 {sp}% 매도")
    print(f"  최종 익절 : ${tp_final}  →  전량 매도  ({tp_label})")
    print(f"  추적 손절 : 신고점 후 ATR×{trail_mult} 하락 시 전량")
    if trail_stop:
        print(f"             현재 기준 초기값: ${trail_stop}")

    # 손절 & 재진입
    print(f"\n{DASH}")
    print(f"  🛑 손절 & 재진입")
    print(f"{DASH}")
    if stop_loss:
        print(f"  손절 트리거 : 종가 ${stop_loss} 이탈 시 즉시 전량 매도")
        print(f"              (장중 일시 이탈 무시 — 종가 기준)")
    print(f"  재진입      : 손절 후 2주 대기 → quant 재실행 후 상위권이면 재매수")
    print(f"  퇴출        : quant 순위 이탈 시 익절 여부 무관하게 매도")

    # 모니터링
    print(f"\n{DASH}")
    print(f"  🔄 모니터링 & 리밸런싱")
    print(f"{DASH}")
    print(f"  보유기간 : {params['label']}  |  다음 점검: {rebal}")
    print(f"  주기     : {params['monitor']}")
    print(f"  점검     : ① quant 순위  ② 실적 후 EPS/매출  ③ 섹터 뉴스")
    print(f"  매도신호 : 순위이탈 / 손절선이탈 / 펀더멘털 훼손")

    # 시나리오 분석 (스트레스 테스트)
    if beta is not None:
        print(f"\n{DASH}")
        print(f"  📉 시나리오 분석  (Beta={beta:.2f} 기반 — 시장 충격 시 예상 손익)")
        print(f"     ※ Beta란? 시장(SPY)이 1% 변할 때 이 주식이 {beta:.1f}% 변하는 경향")
        print(f"{DASH}")
        scenarios = [
            (-0.20, '약세장 −20%',  '금융위기 수준'),
            (-0.10, '조정장 −10%',  '일반적 조정'),
            (-0.05, '소폭 하락 −5%','단기 조정'),
            (+0.05, '소폭 상승 +5%','정상 상승'),
            (+0.10, '강세 랠리+10%','모멘텀 장'),
        ]
        print(f"  {'시나리오':<16} {'예상 등락':>9}  {'예상가':>8}  {'포지션 손익':>13}")
        print(f"  {'─'*15} {'─'*9} {'─'*8} {'─'*13}")
        for mkt_chg, label, _ in scenarios:
            stock_chg = mkt_chg * beta
            new_price = round(price * (1 + stock_chg), 2)
            pnl = round((new_price - price) * total_shares, 2)
            pct = round(stock_chg * 100, 1)
            icon = '📈' if mkt_chg > 0 else '📉'
            print(f"  {icon} {label:<14} {pct:>+8.1f}%  ${new_price:>7.2f}  ${pnl:>+12,.0f}")
        if stop_loss and beta != 0:
            mkt_drop_for_stop = round((stop_loss / price - 1) / beta * 100, 1)
            print(f"\n  ⚠️  손절 도달 조건: 시장이 약 {mkt_drop_for_stop:.1f}% 하락하면 손절가(${stop_loss}) 도달")

    # ── Go/No-Go 판정 (눈에 띄게) ─────────────────────────────────
    signals, score, verdict, vemoji = _go_no_go(
        rsi_val, macd_bull, rr, regime, vix_val,
        days_until_earnings, pos52, hk_pct
    )
    total_signals = len(signals)
    print(f"\n  {BORDER}")
    print(f"  {vemoji}  최종 판정:  {verdict}")
    print(f"     종합 점수: {score:+d}  (총 {total_signals}개 신호 분석)")
    print(f"  {BORDER}")
    print(f"  {'신호':<26} {'판단':>6}  세부 내용")
    print(f"  {'─'*62}")
    for sig_name, sig_val, sig_detail in signals:
        icon = '▲ 긍정' if sig_val > 0 else '▼ 부정' if sig_val < 0 else '─ 중립'
        print(f"  {sig_name:<26} {icon}  {sig_detail}")
    print(f"  {BORDER}")

    # ── 매수 전 체크리스트 ─────────────────────────────────────────
    print(f"\n{DASH}")
    print(f"  ✅ 매수 전 개인 체크리스트")
    print(f"     (판정과 관계없이 스스로 확인해야 할 질문들)")
    print(f"{DASH}")
    checks = []
    if days_until_earnings is not None and 0 <= days_until_earnings <= 14:
        checks.append(('⚠️', f'실적이 {days_until_earnings}일 후입니다 — 결과 예측 가능하신가요?'))
    else:
        checks.append(('☐', '다음 실적 발표 날짜를 알고 있나요?'))
    if rsi_val and rsi_val > 70:
        checks.append(('⚠️', 'RSI 과매수 — 지금 쫓아가는 매수가 아닌지 확인했나요?'))
    else:
        checks.append(('☐', '매수 이유가 단순 상승 추격이 아닌 실질 근거가 있나요?'))
    if rr and rr < 1.5:
        checks.append(('⚠️', f'R/R={rr}:1 — 잠재 수익({upside_pct}%)이 손실({risk_pct}%)보다 충분히 큰가요?'))
    else:
        checks.append(('☐', '손절가 도달 시 감정적으로 버틸 수 있나요?'))
    if pos52 and pos52 > 80:
        checks.append(('⚠️', f'52주 고점({pos52}%) — 고점에서 매수하는 이유가 있나요?'))
    else:
        checks.append(('☐', '이 종목이 내 포트폴리오에서 몇 %인지 계산했나요?'))
    checks.append(('☐', '손절 후 2주 관망하고 재진입 원칙을 지킬 자신이 있나요?'))
    for icon, question in checks:
        print(f"  {icon}  {question}")

    # ── 손실 회복 분석 ─────────────────────────────────────────────
    if stop_loss and risk_pct:
        print(f"\n{DASH}")
        print(f"  🔄 손실 회복 분석  (손절 후 심리 준비)")
        print(f"{DASH}")
        loss_pct = risk_pct
        recovery_needed = round(loss_pct / (1 - loss_pct / 100) if loss_pct < 100 else 0, 1)
        print(f"  손절 시 손실  : −{loss_pct}%")
        print(f"  본전 회복 위해: 손절 후 +{recovery_needed}% 상승 필요")
        print(f"                 ※ 10% 잃으면 11.1%를 벌어야 본전이 되는 구조")
        if vol_val:
            days_to_recover = round(recovery_needed / (vol_val / 252**0.5), 0)
            print(f"  통계적 회복   : 연간변동성 {vol_val}% 기준 약 {int(days_to_recover)}거래일")
            print(f"                 ※ 변동성 기준 평균 소요. 추세에 따라 크게 달라짐")

    print(f"\n  ⚠️  참고용이며 투자 권유가 아닙니다.")
    print(f"     데이터: yfinance  |  기준일: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"     용어 설명 → 메뉴에서 모드 4 선택")
    print(f"{SEP}\n")


# ── 포트폴리오 구성 ───────────────────────────────────────────────────────────

def build_portfolio(capital, sectors=None):
    if sectors is None:
        sectors = list(WATCHLIST.keys())

    print(f"\n{SEP}")
    print(f"  💼  포트폴리오 구성  |  투자금: ${capital:,.0f}")
    print(f"  섹터: {', '.join(sectors)}")
    print(SEP)

    mkt      = _get_market_context()
    regime   = mkt['regime']
    vix_val  = mkt['vix']
    vix_mult = 0.5 if (vix_val is not None and vix_val >= 30) else 1.0
    eff_cap  = capital * vix_mult

    vix_disp = str(vix_val) if vix_val is not None else 'N/A'
    print(f"  🌍 시장: {REGIME_KR[regime]}  |  VIX: {vix_disp}", end='')
    if mkt['spy_price'] and mkt['ma200']:
        spy_vs = round((mkt['spy_price'] / mkt['ma200'] - 1) * 100, 1)
        print(f"  |  SPY vs 200MA: {spy_vs:+.1f}%", end='')
    print()
    if vix_mult < 1.0:
        print(f"  ⚠️  VIX≥30: 배분 50% 자동 축소  (유효 ${eff_cap:,.0f})")

    print("\n  데이터 수집 중...\n")

    tickers = []
    for s in sectors:
        if s in WATCHLIST:
            tickers.extend(WATCHLIST[s])

    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period='1y')
            if hist.empty or len(hist) < 60:
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            close = hist['Close']
            price = round(float(close.iloc[-1]), 2)
            mom_3m  = (round((close.iloc[-1] / close.iloc[-63] - 1) * 100, 1)
                       if len(close) >= 63 else None)
            rsi_val = _rsi(close)
            atr_val = _atr(hist)
            vol_val = _ann_vol(close)
            ma200    = close.rolling(200).mean()
            above_ma = (price > float(ma200.iloc[-1])
                        if len(close) >= 200 and pd.notna(ma200.iloc[-1]) else None)
            sector_name = next((s for s, ts in WATCHLIST.items() if ticker in ts), 'N/A')

            score = 0
            if mom_3m is not None:
                score += min(40, max(0, int(mom_3m + 20)))
            if rsi_val:
                if 40 <= rsi_val <= 65:
                    score += 30
                elif 30 <= rsi_val < 40 or 65 < rsi_val <= 75:
                    score += 15
            if above_ma:
                score += 30

            results.append({'ticker': ticker, 'sector': sector_name, 'price': price,
                             'mom_3m': mom_3m, 'rsi': rsi_val, 'above_ma': above_ma,
                             'atr': atr_val, 'vol': vol_val, 'score': score})
            print(f"  ✓ {ticker:<6} 점수: {score}/100")
        except Exception:
            print(f"  ✗ {ticker:<6} 오류 — 건너뜀")

    if not results:
        print("  ❌ 데이터를 불러올 수 없습니다.")
        return []

    results.sort(key=lambda x: x['score'], reverse=True)

    selected, sector_count = [], {}
    for r in results:
        s = r['sector']
        if sector_count.get(s, 0) < 2 and len(selected) < 6:
            selected.append(r)
            sector_count[s] = sector_count.get(s, 0) + 1

    n         = len(selected)
    alloc_per = eff_cap / n if n > 0 else 0

    print(f"\n{DASH}")
    print(f"  📊 포트폴리오 추천 — 상위 {n}종목")
    print(f"{DASH}")
    print(f"  {'티커':<7} {'섹터':<9} {'현재가':>8} {'3M수익':>8} {'RSI':>6} "
          f"{'200MA':>7} {'변동성':>7} {'배분':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*10}")
    for r in selected:
        ma_str  = '위✅' if r['above_ma'] else '아래❌' if r['above_ma'] is not None else 'N/A'
        mom_str = f"{r['mom_3m']:+.1f}%" if r['mom_3m'] is not None else 'N/A'
        vol_str = f"{r['vol']:.1f}%" if r['vol'] else 'N/A'
        print(f"  {r['ticker']:<7} {r['sector']:<9} ${r['price']:>7.2f} {mom_str:>8} "
              f"{r['rsi']:>6.1f} {ma_str:>7} {vol_str:>7} ${alloc_per:>9,.0f}")

    alloc_note = (f"  (VIX축소 / 원래 ${capital/n:,.0f})" if vix_mult < 1.0 and n > 0 else "")
    print(f"\n  총 투자금: ${capital:,.0f}  |  종목당 배분: ${alloc_per:,.0f}{alloc_note}")
    print(f"  섹터 구성: " + "  |  ".join(f"{s} {c}종목" for s, c in sector_count.items()))

    concentrated = {s: c for s, c in sector_count.items() if c >= 2}
    if concentrated:
        print(f"\n  ⚠️  섹터 집중 경고")
        for s, c in concentrated.items():
            names = [r['ticker'] for r in selected if r['sector'] == s]
            print(f"     {s}: {', '.join(names)} ({c}종목) — 상관관계 ↑ 동반 하락 위험")

    print(f"\n{DASH}")
    print(f"  🔄 포트폴리오 관리 기준")
    print(f"{DASH}")
    print(f"  - 3개월마다 동일비중으로 재조정 (1/{n}씩)")
    print(f"  - 특정 종목 25% 초과 → 즉시 리밸런싱")
    print(f"  - 섹터 비중 최대 40% 유지")
    print(f"  - 분기 실적 후 quant 재실행 → 순위 이탈 종목 교체")

    print(f"\n{DASH}")
    print(f"  📌 추천 종목  (상세 플랜: 모드 1 선택)")
    print(f"{DASH}")
    for i, r in enumerate(selected, 1):
        print(f"  {i}. {r['ticker']:<6}  {r['score']:>3}점  |  {r['sector']}")

    print(f"\n{SEP}\n")
    return [r['ticker'] for r in selected]


# ── 실행 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n{SEP}")
    print(f"  💼  개인 매매 플랜 생성기  v4")
    print(f"  실적경고 · 시장국면 · VIX조절 · Go/No-Go · 펀더멘털 · 시나리오분석 · 체크리스트")
    print(f"{SEP}")
    print(f"\n  모드 선택:")
    print(f"  1. 단일 종목 분석  (티커 → 상세 매매 플랜 + 펀더멘털)")
    print(f"  2. 포트폴리오 구성 (반도체/소프트웨어/헬스케어 자동 추천)")
    print(f"  3. 투자 방법론    (ATR·하프켈리·DCA 실사례)")
    print(f"  4. 용어 사전      (RSI·ATR·VIX·P/E·Beta 등 쉬운 설명)")

    mode = input("\n선택 (1/2/3/4): ").strip()

    if mode == '3':
        show_methodology()
    elif mode == '4':
        show_glossary()
    elif mode == '2':
        capital_raw = input("💵 총 투자 금액 (USD, 예: 30000): $").strip().replace(',', '')
        try:
            capital_input = float(capital_raw)
            if capital_input <= 0:
                raise ValueError
        except ValueError:
            print("⚠️  올바른 금액을 입력해주세요.")
            exit()

        print(f"\n  섹터 선택")
        print(f"  1. 전체  2. 반도체만  3. 소프트웨어만  4. 헬스케어만")
        sec_choice = input("  선택 (1~4, 기본 1): ").strip() or '1'
        sec_map = {'1': None, '2': ['반도체'], '3': ['소프트웨어'], '4': ['헬스케어']}
        picked = build_portfolio(capital_input, sec_map.get(sec_choice, None))

        if picked:
            go = input(f"\n  개별 상세 플랜 종목 (티커 입력, 없으면 Enter): ").upper().strip()
            if go in picked:
                holding_choice = input("  보유기간 (1m/3m/6m, 기본 3m): ").strip() or '3m'
                if holding_choice not in HOLDING_PARAMS:
                    holding_choice = '3m'
                analyze_portfolio(go, capital_input / len(picked), holding_choice)
    else:
        ticker_input = input("\n🔍 종목 티커 (예: AAPL, NVDA): ").upper().strip()
        if not ticker_input:
            print("⚠️  티커를 입력해주세요.")
            exit()

        capital_raw = input("💵 투자 금액 (USD, 예: 10000): $").strip().replace(',', '')
        try:
            capital_input = float(capital_raw)
            if capital_input <= 0:
                raise ValueError
        except ValueError:
            print("⚠️  올바른 금액을 입력해주세요.")
            exit()

        print(f"\n  보유기간:")
        print(f"  1m — 단기(1개월): ATR×1.5 손절, 2회 분할, +8%/+15% 익절")
        print(f"  3m — 중기(3개월): ATR×2.0 손절, 3회 분할, +10%/+20% 익절")
        print(f"  6m — 장기(6개월): ATR×3.0 손절, 4회 분할, +20%/+40% 익절")
        holding_input = input("  입력 (1m/3m/6m, 기본 3m): ").strip() or '3m'
        if holding_input not in HOLDING_PARAMS:
            holding_input = '3m'

        analyze_portfolio(ticker_input, capital_input, holding_input)
