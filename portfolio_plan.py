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

# atr_mult=2.0 모든 기간 고정 (터틀 트레이더 표준)
# tp_fallback_pct: 월가 목표가 없을 때 쓸 자체 익절 폭
# entry_end_drop: 진입 하단 (현재가 대비 %)
# rr_min: 최소 허용 R/R 비율 (EV 판단 기준)
# regime_fit: 이 전략이 유효한 시장 국면
HOLDING_PARAMS = {
    '1m': {
        'label': '단기 스윙 (1개월)',
        'atr_mult': 2.0, 'tp_fallback_pct': 0.10, 'entry_end_drop': -0.05,
        'monitor': '매일 종가', 'rebal_months': 1, 'rr_min': 1.5,
        'note': '모멘텀 강할 때 단기 차익. 실적 발표 전후 포지션 주의.',
        'regime_fit': ('bullish', 'neutral'),
    },
    '3m': {
        'label': '중기 포지션 (3개월)',
        'atr_mult': 2.0, 'tp_fallback_pct': 0.20, 'entry_end_drop': -0.08,
        'monitor': '주 1회', 'rebal_months': 3, 'rr_min': 2.0,
        'note': '분기 실적 사이클 활용. 분할 진입으로 리스크 분산.',
        'regime_fit': ('bullish', 'neutral', 'bearish'),
    },
    '6m': {
        'label': '장기 성장 (6개월)',
        'atr_mult': 2.0, 'tp_fallback_pct': 0.35, 'entry_end_drop': -0.12,
        'monitor': '월 1회', 'rebal_months': 6, 'rr_min': 2.5,
        'note': '성장 스토리 베팅. 강세장 + 강한 모멘텀 구간에 적합.',
        'regime_fit': ('bullish',),
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
    # Faber(2007) + VIX overlay: 강세장 베이스 0.58, 약세장 0.48
    wr = 0.55
    if regime == 'bullish':
        wr += 0.03
    elif regime == 'bearish':
        wr -= 0.07
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


def _period_fit(period_key, regime, vix_val, rsi_val, mom_12, rr):
    """보유기간별 적합도 점수 (0~100). 근거: Faber 2007 + J&T 1993 + VIX overlay."""
    p = HOLDING_PARAMS[period_key]
    score = 50  # baseline

    # 1. 시장 국면 적합도 (Faber 2007: 200MA 기반 국면)
    if regime in p['regime_fit']:
        score += 20
    else:
        score -= 20

    # 2. VIX 리스크 (단기일수록 VIX 민감)
    if vix_val is not None:
        if vix_val >= 30:
            penalty = {'1m': -20, '3m': -10, '6m': -5}
            score += penalty.get(period_key, -10)
        elif vix_val >= 20:
            penalty = {'1m': -8, '3m': -4, '6m': 0}
            score += penalty.get(period_key, -4)
        else:
            bonus = {'1m': 10, '3m': 5, '6m': 0}
            score += bonus.get(period_key, 5)

    # 3. 모멘텀 (J&T 1993: 12-1개월 모멘텀. 장기 전략에 더 중요)
    if mom_12 is not None:
        if mom_12 > 20:
            bonus = {'1m': 5, '3m': 10, '6m': 15}
            score += bonus.get(period_key, 10)
        elif mom_12 > 0:
            bonus = {'1m': 3, '3m': 6, '6m': 8}
            score += bonus.get(period_key, 6)
        else:
            bonus = {'1m': 0, '3m': -5, '6m': -15}
            score += bonus.get(period_key, -5)

    # 4. RSI (단기는 과매수 회피, 장기는 덜 민감)
    if rsi_val is not None:
        if rsi_val > 75:
            penalty = {'1m': -15, '3m': -8, '6m': -3}
            score += penalty.get(period_key, -8)
        elif 45 <= rsi_val <= 65:
            score += 5

    # 5. R/R 비율 (최소 rr_min 미달 시 감점)
    if rr is not None:
        if rr >= p['rr_min']:
            score += 10
        elif rr >= 1.5:
            score += 0
        else:
            score -= 15

    return max(0, min(100, score))


def _get_upcoming_events(info, raw_dividends, earnings_date, days_ahead=90):
    """거시경제 + 실적 + 배당 이벤트를 날짜순으로 반환."""
    try:
        from step3_news import MACRO_CALENDAR
    except ImportError:
        MACRO_CALENDAR = []

    now    = pd.Timestamp.now().normalize()
    cutoff = now + pd.Timedelta(days=days_ahead)
    events = []

    # 거시경제 이벤트
    for row in MACRO_CALENDAR:
        dt = pd.Timestamp(row[0])
        if now <= dt <= cutoff:
            diff = (dt - now).days
            events.append({
                'date': dt, 'dday': diff,
                'name': row[1], 'cat': row[3], 'icon': row[4],
                'scope': '거시',
            })

    # EPS 실적 발표
    if earnings_date is not None:
        dt = pd.Timestamp(earnings_date).normalize()
        if now <= dt <= cutoff:
            diff = (dt - now).days
            events.append({
                'date': dt, 'dday': diff,
                'name': '실적 발표 (EPS)', 'cat': '실적', 'icon': '📢',
                'scope': '종목',
            })

    # 배당 ex-date
    try:
        ex_raw = info.get('exDividendDate')
        if ex_raw:
            ex_dt = pd.Timestamp(ex_raw, unit='s').normalize()
            if now <= ex_dt <= cutoff:
                diff = (ex_dt - now).days
                events.append({
                    'date': ex_dt, 'dday': diff,
                    'name': '배당락일 (Ex-Div)', 'cat': '배당', 'icon': '💰',
                    'scope': '종목',
                })
    except Exception:
        pass

    # 배당 지급 예상일 (최근 4회 간격 평균)
    try:
        if raw_dividends is not None and len(raw_dividends) >= 4:
            dates = raw_dividends.index.sort_values()[-5:]
            gaps  = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
            avg_gap = int(sum(gaps) / len(gaps))
            last_pay = pd.Timestamp(dates[-1]).normalize()
            next_pay = last_pay + pd.Timedelta(days=avg_gap)
            if now <= next_pay <= cutoff:
                diff = (next_pay - now).days
                events.append({
                    'date': next_pay, 'dday': diff,
                    'name': '배당 지급 예상', 'cat': '배당', 'icon': '💵',
                    'scope': '종목',
                })
    except Exception:
        pass

    events.sort(key=lambda x: x['date'])
    return events


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
          "★ 모든 기간 ATR×2.0 통일: 연구 결과 1.5~2.5 범위에서 2.0이",
          "  샤프비율·최대손실비율 균형 기준 최적 (Kaufman, 2013)",
          "리스크: 횡보장에서 잦은 손절 → 수수료·슬리피지 누적"]),
        ("2. 기대값(EV) 기반 리스크/보상",
         ["실사례: 레이 달리오 All-Weather / 르네상스 테크",
          "  → 단순 R/R이 아닌 확률가중 기대값으로 포지션 결정",
          "원리  : EV = 승률×수익 − 패율×손실",
          "       EV > 0 이고 R/R ≥ rr_min이어야 진입",
          "       손실 비대칭 고려: -10% 손실 → +11.1% 회복 필요",
          "★ 손절 임박 시 EV 급감 → 기계적 청산이 심리적 버티기보다 유리",
          "리스크: 승률 추정 오류 시 과도 베팅 → 보수적 가정 권장"]),
        ("3. 하프켈리 포지션 사이징 (Half-Kelly Criterion)",
         ["실사례: Ed Thorp (블랙잭 → 헤지펀드, 1960~2000년대)",
          "  → Kelly 공식으로 수십억 달러 운용, 230개월 무손실",
          "  → 르네상스 테크놀로지, DE Shaw 등 퀀트 펀드 적용",
          "원리  : 수학적 최적 베팅의 절반만 사용 → 파산 리스크 감소",
          "       f = (승률×수익 − 패율×손실) ÷ 수익",
          "★ 고정 승률 → 국면+VIX 기반 동적 계산",
          "       강세+VIX안정 → 0.58 / 약세+VIX공포 → 0.43",
          "리스크: 승률 추정 오류 시 과도 베팅 → 보수적 가정 권장"]),
        ("4. 모멘텀 팩터 (J&T 1993) + Faber 트렌드",
         ["Jegadeesh & Titman(1993): 12-1개월 모멘텀 포트폴리오",
          "  → 과거 수익 상위 종목이 향후 3~12개월 지속 아웃퍼폼",
          "  → 연 평균 초과수익 +1%/월 (거래비용 전 기준)",
          "Faber(2007): 200MA 기반 추세추종",
          "  → SPY 200MA 위: 주식 보유 / 아래: 현금. S&P 장기 성과 개선",
          "★ 두 팩터 결합: 모멘텀 강 + 200MA 위 = 최고 승률 조합",
          "  Aronson(2006) Evidence-Based TA: 단일 신호보다 컨플루언스 중요"]),
    ]

    for title, lines in sections:
        print(f"\n{DASH}")
        print(f"  {title}")
        print(DASH)
        for line in lines:
            print(f"  {line}")

    print(f"\n{DASH}")
    print(f"  ※ 조합 (이 툴의 접근법)")
    print(DASH)
    print(f"  ATR×2 손절 + EV 기반 R/R + 하프켈리 비중 + 모멘텀+트렌드 진입의 조합은")
    print(f"  체계적 트레이더들이 실전에서 가장 많이 사용하는 리스크 관리 체계입니다.")
    print(f"  과거 성과가 미래를 보장하지 않으며 개인 판단이 최우선입니다.")
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

    # 배당 이력
    try:
        raw_dividends = stock.dividends
        if raw_dividends.index.tz is not None:
            raw_dividends.index = raw_dividends.index.tz_convert(None)
    except Exception:
        raw_dividends = None

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
    # 12-1개월 모멘텀 (J&T 1993)
    mom_12 = (round((float(close.iloc[-1]) / float(close.iloc[-252]) - 1) * 100, 1)
              if len(close) >= 252 else
              round((float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100, 1))

    pos52      = (round((price - l52) / (h52 - l52) * 100, 1)
                  if (h52 and l52 and h52 != l52) else None)
    upside_pct = round((target - price) / price * 100, 1) if target else None

    atr_mult = params['atr_mult']  # 항상 2.0

    stop_loss = round(price - atr_mult * atr_val, 2) if atr_val else None
    risk_pct  = round(atr_mult * atr_val / price * 100, 1) if atr_val else None

    # 익절가: 월가 컨센서스 우선, 없으면 tp_fallback_pct
    if target and upside_pct and upside_pct > 0:
        tp_final  = round(target, 2)
        tp_label  = f'월가 컨센서스 ({n_anal}명 평균)'
        tp_upside = upside_pct
    else:
        tp_final  = round(price * (1 + params['tp_fallback_pct']), 2)
        tp_label  = f"자체 계산 ({int(params['tp_fallback_pct']*100)}% 목표)"
        tp_upside = params['tp_fallback_pct'] * 100

    if upside_pct and risk_pct and risk_pct > 0:
        rr       = round(tp_upside / risk_pct, 2)
        rr_label = '우수 ✅' if rr >= 2.0 else '보통 🟡' if rr >= 1.5 else '주의 🔴'
    else:
        rr = rr_label = None

    # EV 기반 승률 (Aronson 2006 컨플루언스 점수 기반)
    # Go/No-Go 점수를 먼저 일부 계산해 승률 추정
    _pre_score = 0
    if rsi_val and 45 <= rsi_val <= 65: _pre_score += 1
    if macd_bull: _pre_score += 1
    if regime == 'bullish': _pre_score += 1
    elif regime == 'bearish': _pre_score -= 1
    if vix_val and vix_val < 20: _pre_score += 1
    elif vix_val and vix_val >= 30: _pre_score -= 1

    wp = (0.62 if _pre_score >= 4 else 0.58 if _pre_score >= 3 else
          0.54 if _pre_score >= 2 else 0.51 if _pre_score >= 1 else
          0.48 if _pre_score == 0 else 0.44 if _pre_score >= -1 else 0.40)

    if risk_pct and tp_upside and risk_pct > 0 and tp_upside > 0:
        ev           = round(wp * tp_upside - (1 - wp) * risk_pct, 1)
        breakeven_wr = round(risk_pct / (risk_pct + tp_upside) * 100, 1)
        kelly_f      = (wp * (tp_upside / 100) - (1 - wp) * (risk_pct / 100)) / (tp_upside / 100)
        hk_f         = max(0.0, min(kelly_f / 2, 1.0))
        hk_pct       = round(hk_f * 100, 1)
        hk_amt       = round(effective_capital * hk_f, 0)
    else:
        ev = breakeven_wr = hk_pct = hk_amt = None

    w = _dynamic_kelly_wr(regime, vix_val)

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
    print(f"  🌍 시장 환경")
    print(f"{DASH}")
    if mkt['spy_price'] and mkt['ma200']:
        spy_vs_ma = round((mkt['spy_price'] / mkt['ma200'] - 1) * 100, 1)
        print(f"  SPY / 200MA  : ${mkt['spy_price']} / ${mkt['ma200']}  ({spy_vs_ma:+.1f}%)")
    print(f"  시장 국면    : {REGIME_KR[regime]}")
    if vix_val is not None:
        vix_lbl = ('공포구간 🔴 → 포지션 50% 자동 축소' if vix_val >= 30
                   else '주의구간 🟡' if vix_val >= 20 else '안정구간 🟢')
        print(f"  VIX (공포지수): {vix_val}  →  {vix_lbl}")
        if vix_val >= 30:
            print(f"  ⚠️  유효 투자금: ${effective_capital:,.0f}  (원래 ${capital:,.0f}의 50%)")
    print(f"  하프켈리 승률: {w:.2f}  (국면+VIX 동적 계산)")

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
        pos52_lbl = ('고점권 ⚠️' if pos52 > 75 else '저점권 💡' if pos52 < 25 else '중간')
        print(f"  52주 위치    : {pos52}%  →  {pos52_lbl}")
    rsi_lbl = ('과매수 ⚠️' if rsi_val > 70 else '과매도 💡' if rsi_val < 30 else '중립')
    print(f"  RSI(14)      : {rsi_val}  →  {rsi_lbl}")
    print(f"  ATR(14)      : ${atr_val}  ← 손절 계산 기준")
    print(f"  연간 변동성  : {vol_val}%")
    print(f"  12M 모멘텀   : {mom_12:+.1f}%  (J&T 1993 기준)")
    macd_lbl = ('상승 모멘텀 ▲' if macd_bull else '하락 모멘텀 ▼')
    print(f"  MACD         : {macd_lbl}")
    if target:
        range_str = f"  범위 ${t_low:.2f}~${t_high:.2f}" if (t_low and t_high) else ""
        print(f"  월가 목표가  : ${target:.2f}  ({upside_pct:+.1f}%){('  |  ' + range_str) if range_str else ''}")

    # 펀더멘털 지표
    print(f"\n{DASH}")
    print(f"  💹 펀더멘털 지표")
    print(f"{DASH}")
    has_fund = False
    if pe_trail is not None and pe_trail > 0:
        print(f"  P/E (현재)   : {pe_trail:.1f}배")
        has_fund = True
    if pe_fwd is not None and pe_fwd > 0:
        print(f"  P/E (예상)   : {pe_fwd:.1f}배")
        has_fund = True
    if peg is not None and peg > 0:
        peg_lbl = '저평가 💡' if peg < 1.0 else '적정' if peg <= 1.5 else '주의 ⚠️'
        print(f"  PEG 비율     : {peg:.2f}  →  {peg_lbl}")
        has_fund = True
    if beta is not None:
        beta_lbl = ('저변동 방어주' if beta < 0.8 else
                    '시장 동조' if beta <= 1.2 else '고변동 공격주')
        print(f"  Beta         : {beta:.2f}  →  {beta_lbl}")
        has_fund = True
    if de_ratio is not None:
        de_lbl = ('부채 낮음 💡' if de_ratio < 50 else '적정' if de_ratio < 150 else '부채 높음 ⚠️')
        print(f"  부채비율(D/E): {de_ratio:.0f}%  →  {de_lbl}")
        has_fund = True
    if fcf_yield is not None:
        fy_lbl = '우수 💡' if fcf_yield > 5 else '양호' if fcf_yield > 2 else '낮음'
        print(f"  FCF Yield    : {fcf_yield:.1f}%  →  {fy_lbl}")
        has_fund = True
    if div_yield is not None and div_yield > 0:
        print(f"  배당수익률   : {div_yield:.2f}%")
        has_fund = True
    if not has_fund:
        print(f"  (펀더멘털 데이터 없음 — ETF 또는 데이터 미제공)")

    # 보유기간별 적합도 비교
    print(f"\n{DASH}")
    print(f"  📅 보유기간별 적합도  (선택: {params['label']})")
    print(f"     근거: Faber(2007) 국면 + J&T(1993) 모멘텀 + VIX overlay")
    print(f"{DASH}")
    print(f"  {'구분':<18} {'단기(1개월)':>12} {'중기(3개월)':>12} {'장기(6개월)':>12}")
    print(f"  {'─'*17} {'─'*12} {'─'*12} {'─'*12}")

    fits = {k: _period_fit(k, regime, vix_val, rsi_val, mom_12, rr) for k in ['1m', '3m', '6m']}
    fit_labels = {k: ('★ 최적' if fits[k] >= 70 else '양호' if fits[k] >= 55 else '주의 ⚠️')
                  for k in fits}
    stops  = {k: (round(price - HOLDING_PARAMS[k]['atr_mult'] * atr_val, 2) if atr_val else None)
              for k in ['1m', '3m', '6m']}
    tp_fbs = {k: round(price * (1 + HOLDING_PARAMS[k]['tp_fallback_pct']), 2) for k in ['1m', '3m', '6m']}
    tp_fin = {k: (round(target, 2) if target and upside_pct and upside_pct > 0 else tp_fbs[k])
              for k in ['1m', '3m', '6m']}

    def _stop_str(k): return f"${stops[k]}" if stops[k] else 'N/A'
    def _tp_str(k):   return f"${tp_fin[k]}"

    print(f"  {'손절가 (ATR×2.0)':<18} {_stop_str('1m'):>12} {_stop_str('3m'):>12} {_stop_str('6m'):>12}")
    print(f"  {'익절가 (목표가)':<18} {_tp_str('1m'):>12} {_tp_str('3m'):>12} {_tp_str('6m'):>12}")
    print(f"  {'모니터링':<18} {'매일':>12} {'주 1회':>12} {'월 1회':>12}")
    print(f"  {'적합도 점수':<18} {fits['1m']:>11}점 {fits['3m']:>11}점 {fits['6m']:>11}점")
    print(f"  {'평가':<18} {fit_labels['1m']:>12} {fit_labels['3m']:>12} {fit_labels['6m']:>12}")

    # 진입 범위 (분할 매수)
    entry_start = price
    entry_end   = round(price * (1 + params['entry_end_drop']), 2)
    print(f"\n{DASH}")
    print(f"  📥 분할 매수 진입 범위")
    print(f"{DASH}")
    print(f"  진입 시작    : ${entry_start}  (현재가 즉시 진입 기준)")
    print(f"  진입 하단    : ${entry_end}  ({int(abs(params['entry_end_drop']*100))}% 하락 시 추가 매수 하단)")
    print(f"  ※ 분할 횟수와 비율은 투자자 판단 — 추천: 현재가에서 시작해 하락 시 점진 추가")
    if stop_loss:
        print(f"  손절가 (고정): ${stop_loss}  (ATR×2.0, 종가 기준 이탈 시 청산)")
    print(f"  익절가 목표  : ${tp_final}  ({tp_label})")

    # SPY 상대 수익률
    if stock_ret_20d is not None and spy_ret_20d is not None:
        rel = round(stock_ret_20d - spy_ret_20d, 1)
        print(f"\n  📊 SPY 상대 성과  (최근 20일)")
        print(f"  {ticker:<6} {stock_ret_20d:+.1f}%  /  SPY {spy_ret_20d:+.1f}%  /  초과: {rel:+.1f}%")
        if stock_ret_20d < 0 and spy_ret_20d < 0 and abs(rel) < 3:
            print(f"  → 시장 전반 하락 동조 — 분할매수 유효")
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

    # 리스크 / 보상 — EV 기반
    print(f"\n{DASH}")
    print(f"  📊 리스크 / 보상 — 기대값(EV) 분석")
    print(f"     근거: Aronson(2006) 컨플루언스 기반 승률 추정 + Half-Kelly")
    print(f"{DASH}")
    if rr and ev is not None:
        ev_lbl  = '양호 ✅' if ev > 3 else '보통 🟡' if ev > 0 else '음수 🔴 진입 비추천'
        rr_req  = '충족 ✅' if rr >= params['rr_min'] else f"미달 🔴 (최소 {params['rr_min']}:1 필요)"
        print(f"  R/R 비율     : {rr}:1  →  {rr_label}  ({rr_req})")
        print(f"  예상 수익    : +{tp_upside:.1f}%  ({tp_label})")
        print(f"  최대 손실    : −{risk_pct}%  (ATR×2.0 손절)")
        print(f"  추정 승률    : {wp:.0%}  (컨플루언스 신호 {_pre_score:+d}점 기준)")
        print(f"  기대값(EV)   : {ev:+.1f}%  →  {ev_lbl}")
        print(f"  손익분기 승률: {breakeven_wr}%  (이 이상 맞아야 플러스EV)")
    if hk_pct is not None:
        k_lbl = ('적정' if 0 < hk_pct <= 25 else
                 '과도 — 축소 권장' if hk_pct > 25 else '진입 비추천 (기대값 음수)')
        print(f"  하프켈리 비중: {hk_pct}%  (${hk_amt:,.0f})  →  {k_lbl}")

    # 시나리오 분석 (Beta 기반)
    if beta is not None:
        print(f"\n{DASH}")
        print(f"  📉 베타 시나리오  (Beta={beta:.2f} — 시장 충격 시 예상 손익)")
        print(f"{DASH}")
        scenarios = [
            (-0.20, '약세장 −20%',  '금융위기 수준'),
            (-0.10, '조정장 −10%',  '일반적 조정'),
            (-0.05, '소폭 하락 −5%', '단기 조정'),
            (+0.05, '소폭 상승 +5%', '정상 상승'),
            (+0.10, '강세 랠리+10%', '모멘텀 장'),
        ]
        print(f"  {'시나리오':<16} {'예상 등락':>9}  {'예상가':>8}  {'포지션 손익':>13}")
        print(f"  {'─'*15} {'─'*9} {'─'*8} {'─'*13}")
        est_shares = max(1, int(effective_capital // price))
        for mkt_chg, label, _ in scenarios:
            stock_chg = mkt_chg * beta
            new_price = round(price * (1 + stock_chg), 2)
            pnl = round((new_price - price) * est_shares, 2)
            pct = round(stock_chg * 100, 1)
            icon = '📈' if mkt_chg > 0 else '📉'
            print(f"  {icon} {label:<14} {pct:>+8.1f}%  ${new_price:>7.2f}  ${pnl:>+12,.0f}")
        if stop_loss and beta != 0:
            mkt_drop_for_stop = round((stop_loss / price - 1) / beta * 100, 1)
            print(f"\n  ⚠️  손절 조건: 시장 약 {mkt_drop_for_stop:.1f}% 하락 시 손절가(${stop_loss}) 도달")

    # Go/No-Go 판정
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

    # 모니터링 & 이벤트 캘린더
    print(f"\n{DASH}")
    print(f"  🔄 모니터링 & 이벤트 캘린더  (90일 이내)")
    print(f"{DASH}")
    print(f"  보유기간 : {params['label']}  |  다음 점검: {rebal}")
    print(f"  주기     : {params['monitor']}")
    print(f"  매도신호 : 순위이탈 / 손절선이탈 / 펀더멘털 훼손")

    events = _get_upcoming_events(info, raw_dividends, earnings_date, days_ahead=90)
    if events:
        print(f"\n  {'날짜':<12} {'D-Day':>6}  {'카테고리':<10} {'이벤트':<24} {'범위':<6}")
        print(f"  {'─'*11} {'─'*6}  {'─'*9} {'─'*23} {'─'*5}")
        for ev in events:
            dday_str = f"D-{ev['dday']}" if ev['dday'] > 0 else "D-Day"
            scope_str = '종목' if ev['scope'] == '종목' else '거시'
            warn = ' ⚠️' if ev['dday'] <= 7 else ''
            print(f"  {ev['date'].strftime('%Y-%m-%d'):<12} {dday_str:>6}  "
                  f"{ev['icon']} {ev['cat']:<8} {ev['name']:<24} {scope_str}{warn}")
    else:
        print(f"  (90일 이내 주요 이벤트 없음)")

    print(f"\n  ⚠️  참고용이며 투자 권유가 아닙니다.")
    print(f"     데이터: yfinance  |  기준일: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
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

            close   = hist['Close']
            price   = round(float(close.iloc[-1]), 2)
            # J&T 1993: 12-1개월 모멘텀
            mom_12  = (round((close.iloc[-1] / close.iloc[-252] - 1) * 100, 1)
                       if len(close) >= 252 else
                       round((close.iloc[-1] / close.iloc[0] - 1) * 100, 1))
            # 3개월 모멘텀 (단기 확인)
            mom_3   = (round((close.iloc[-1] / close.iloc[-63] - 1) * 100, 1)
                       if len(close) >= 63 else None)
            rsi_val = _rsi(close)
            atr_val = _atr(hist)
            vol_val = _ann_vol(close)
            ma200   = close.rolling(200).mean()
            ma50    = close.rolling(50).mean()
            above_ma  = (price > float(ma200.iloc[-1])
                         if len(close) >= 200 and pd.notna(ma200.iloc[-1]) else None)
            above_ma50 = (price > float(ma50.iloc[-1])
                          if len(close) >= 50 and pd.notna(ma50.iloc[-1]) else None)
            ma_slope  = (float(ma200.iloc[-1]) > float(ma200.iloc[-20])
                         if len(close) >= 200 and pd.notna(ma200.iloc[-20]) else None)

            # SPY 3개월 수익률 대비 상대 강도
            spx_3m = None
            try:
                sp_hist = yf.Ticker('^GSPC').history(period='3mo')
                if not sp_hist.empty:
                    spx_3m = round(
                        (float(sp_hist['Close'].iloc[-1]) / float(sp_hist['Close'].iloc[0]) - 1) * 100, 1)
            except Exception:
                pass

            sector_name = next((s for s, ts in WATCHLIST.items() if ticker in ts), 'N/A')

            # ── 다중 팩터 스코어링 (총 100점) ──────────────────────────────
            # 1. 추세 정렬 40점 (Faber 2007: 200MA 기반)
            trend_score = 0
            if above_ma:
                trend_score += 22
            if above_ma50:
                trend_score += 10
            if ma_slope:
                trend_score += 8

            # 2. 모멘텀 팩터 35점 (J&T 1993: 12-1개월)
            mom_12_score = max(0, min(20, int((mom_12 + 10) / 4.5))) if mom_12 is not None else 0
            mom_3_score  = (max(0, min(15, int((mom_3 + 15) / 3))) if mom_3 is not None else 0)

            # 3. 기술적 품질 15점
            tech_score = 0
            if rsi_val:
                if 45 <= rsi_val <= 65:
                    tech_score += 10
                elif (38 <= rsi_val < 45) or (65 < rsi_val <= 72):
                    tech_score += 5
            if _macd_signal(close):
                tech_score += 5

            # 4. SPY 상대 강도 보너스 10점
            rs_bonus = 0
            if spx_3m is not None and mom_3 is not None:
                rel_rs = mom_3 - spx_3m
                if rel_rs > 5:
                    rs_bonus = 10
                elif rel_rs > 0:
                    rs_bonus = 5

            score = trend_score + mom_12_score + mom_3_score + tech_score + rs_bonus

            # 패널티
            if rsi_val and rsi_val > 78:
                score -= 15
            h52 = stock.info.get('fiftyTwoWeekHigh') if hasattr(stock, 'info') else None
            l52 = stock.info.get('fiftyTwoWeekLow')  if hasattr(stock, 'info') else None
            pos52 = (round((price - l52) / (h52 - l52) * 100, 1)
                     if (h52 and l52 and h52 != l52) else None)
            if pos52 and pos52 > 92:
                score -= 8

            score = max(0, min(100, score))

            results.append({'ticker': ticker, 'sector': sector_name, 'price': price,
                             'mom_12': mom_12, 'mom_3': mom_3, 'rsi': rsi_val,
                             'above_ma': above_ma, 'atr': atr_val, 'vol': vol_val,
                             'score': score,
                             'trend': trend_score, 'momentum': mom_12_score + mom_3_score,
                             'tech': tech_score, 'rs': rs_bonus})
            print(f"  ✓ {ticker:<6} 점수: {score:>3}/100  "
                  f"(추세:{trend_score} 모멘텀:{mom_12_score+mom_3_score} 기술:{tech_score} RS:{rs_bonus})")
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
    print(f"     스코어링: 추세40 + 모멘텀35 + 기술15 + SPY-RS10")
    print(f"{DASH}")
    print(f"  {'티커':<7} {'섹터':<9} {'현재가':>8} {'12M':>7} {'3M':>7} {'RSI':>6} "
          f"{'200MA':>7} {'점수':>5} {'배분':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*10}")
    for r in selected:
        ma_str   = '위✅' if r['above_ma'] else '아래❌' if r['above_ma'] is not None else 'N/A'
        mom12str = f"{r['mom_12']:+.1f}%" if r['mom_12'] is not None else 'N/A'
        mom3str  = f"{r['mom_3']:+.1f}%"  if r['mom_3']  is not None else 'N/A'
        print(f"  {r['ticker']:<7} {r['sector']:<9} ${r['price']:>7.2f} {mom12str:>7} {mom3str:>7} "
              f"{r['rsi']:>6.1f} {ma_str:>7} {r['score']:>5} ${alloc_per:>9,.0f}")

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
    print(f"  💼  개인 매매 플랜 생성기  v5")
    print(f"  EV기반R/R · 보유기간적합도 · 이벤트캘린더 · 베타시나리오 · Go/No-Go")
    print(f"{SEP}")
    print(f"\n  모드 선택:")
    print(f"  1. 단일 종목 분석  (티커 → 상세 매매 플랜 + 펀더멘털)")
    print(f"  2. 포트폴리오 구성 (반도체/소프트웨어/헬스케어 자동 추천)")
    print(f"  3. 투자 방법론    (ATR·EV·하프켈리·모멘텀 실사례)")

    mode = input("\n선택 (1/2/3): ").strip()

    if mode == '3':
        show_methodology()
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
        print(f"  1m — 단기(1개월): ATR×2.0 손절, 진입범위 −5%,  R/R최소 1.5:1")
        print(f"  3m — 중기(3개월): ATR×2.0 손절, 진입범위 −8%,  R/R최소 2.0:1")
        print(f"  6m — 장기(6개월): ATR×2.0 손절, 진입범위 −12%, R/R최소 2.5:1")
        holding_input = input("  입력 (1m/3m/6m, 기본 3m): ").strip() or '3m'
        if holding_input not in HOLDING_PARAMS:
            holding_input = '3m'

        analyze_portfolio(ticker_input, capital_input, holding_input)
