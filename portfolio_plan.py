import yfinance as yf
import pandas as pd


# ── 보조 계산 함수 ────────────────────────────────────────────────────────────

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
    """MACD > Signal 이면 True (상승 모멘텀)"""
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
    """0~100% 텍스트 바"""
    filled = int(round(pct / 100 * width))
    return '█' * filled + '░' * (width - filled)


# ── 메인 분석 함수 ────────────────────────────────────────────────────────────

def analyze_portfolio(ticker, capital):
    sep  = '=' * 65
    dash = '─' * 65

    print(f"\n{sep}")
    print(f"  💼  [{ticker}]  개인 매매 플랜  |  투자금: ${capital:,.0f}")
    print(sep)
    print("  데이터 수집 중...")

    stock = yf.Ticker(ticker)
    info  = stock.info
    hist  = stock.history(period='1y')

    if hist.empty or not info:
        print(f"  ❌ [{ticker}] 데이터를 불러올 수 없습니다.")
        return

    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)

    # ── 기본 정보 ──────────────────────────────────────────────
    name     = info.get('longName') or info.get('shortName') or ticker
    sector   = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    price    = round(float(hist['Close'].iloc[-1]), 2)
    h52      = info.get('fiftyTwoWeekHigh')
    l52      = info.get('fiftyTwoWeekLow')
    target   = info.get('targetMeanPrice')
    t_high   = info.get('targetHighPrice')
    t_low    = info.get('targetLowPrice')
    n_anal   = int(info.get('numberOfAnalystOpinions') or 0)
    rec      = info.get('recommendationKey', '')
    short_pct = info.get('shortPercentOfFloat')

    # ── 기술적 지표 ────────────────────────────────────────────
    atr_val  = _atr(hist)
    rsi_val  = _rsi(hist['Close'])
    vol_val  = _ann_vol(hist['Close'])
    macd_bull = _macd_signal(hist['Close'])

    # 52주 위치
    pos52 = round((price - l52) / (h52 - l52) * 100, 1) if (h52 and l52 and h52 != l52) else None

    # 목표가 업사이드
    upside_pct = round((target - price) / price * 100, 1) if target else None

    # ── 손절 계산 ──────────────────────────────────────────────
    stop_loss  = round(price - 2 * atr_val, 2)  if atr_val else None
    risk_pct   = round(2 * atr_val / price * 100, 1) if atr_val else None
    trail_stop = round(price - 1.5 * atr_val, 2) if atr_val else None  # 고점 추적용

    # ── 진입 가격 (3회 분할) ──────────────────────────────────
    e1 = price                        # 즉시
    e2 = round(price * 0.95, 2)       # -5%
    e3 = round(price * 0.90, 2)       # -10% (또는 RSI<30 구간)

    # ── 포지션 사이징 ─────────────────────────────────────────
    # 즉시 전액
    sh_imm   = max(1, int(capital // price))
    cost_imm = round(sh_imm * price, 2)
    ml_imm   = round(sh_imm * 2 * atr_val, 2) if atr_val else None

    # 분할매수 (33/33/34%)
    a1, a2, a3 = capital * 0.33, capital * 0.33, capital * 0.34
    sh1 = max(1, int(a1 // e1))
    sh2 = max(1, int(a2 // e2))
    sh3 = max(1, int(a3 // e3))
    avg_cost = round((sh1*e1 + sh2*e2 + sh3*e3) / (sh1+sh2+sh3), 2)
    ml_dca   = round((sh1+sh2+sh3) * 2 * atr_val, 2) if atr_val else None

    # ── 익절 목표가 ───────────────────────────────────────────
    tp1 = round(price * 1.10, 2)
    tp2 = round(price * 1.20, 2)
    tp3 = round(target, 2) if target else round(price * 1.30, 2)

    # ── R/R 비율 ──────────────────────────────────────────────
    if upside_pct and risk_pct and risk_pct > 0:
        rr = round(upside_pct / risk_pct, 2)
        rr_label = '우수 ✅' if rr >= 2.0 else '보통 🟡' if rr >= 1.5 else '주의 🔴'
    else:
        rr = rr_label = None

    # ── 하프켈리 비중 ─────────────────────────────────────────
    if upside_pct and risk_pct and upside_pct > 0 and risk_pct > 0:
        w       = 0.55   # 가정 승률
        kelly_f = (w * (upside_pct/100) - (1-w) * (risk_pct/100)) / (upside_pct/100)
        hk_f    = max(0.0, min(kelly_f / 2, 1.0))
        hk_pct  = round(hk_f * 100, 1)
        hk_amt  = round(capital * hk_f, 0)
    else:
        hk_pct = hk_amt = None

    # ── 리밸런싱 날짜 ─────────────────────────────────────────
    rebal = (pd.Timestamp.now() + pd.DateOffset(months=3)).strftime('%Y-%m-%d')

    # ══════════════════════════════════════════════════════════
    # 출력
    # ══════════════════════════════════════════════════════════

    # 기본 정보
    print(f"\n  🏢  {name}")
    print(f"      섹터: {sector}  |  산업: {industry}")
    print(f"      애널리스트 {n_anal}명  |  컨센서스: {rec or 'N/A'}")
    if short_pct:
        sc = '#🔴' if short_pct > 0.10 else '🟡' if short_pct > 0.05 else '🟢'
        print(f"      공매도 비율: {sc} {short_pct*100:.1f}%")

    # 현재 상태
    print(f"\n{dash}")
    print(f"  📍 현재 상태")
    print(f"{dash}")
    print(f"  현재가       : ${price}")
    if h52 and l52 and pos52 is not None:
        bar = _bar(pos52)
        print(f"  52주 범위    : ${l52:.2f} {'─'*3} [{bar}] {'─'*3} ${h52:.2f}")
        print(f"  52주 위치    : {pos52}%  ({'고점 근처 ⚠️' if pos52 > 75 else '저점 근처 💡' if pos52 < 25 else '중간'})")
    print(f"  RSI(14)      : {rsi_val}  ({'과매수 ⚠️' if rsi_val > 70 else '과매도 💡' if rsi_val < 30 else '중립'})")
    print(f"  ATR(14)      : ${atr_val}  (일 평균 변동폭)")
    print(f"  연간 변동성  : {vol_val}%")
    print(f"  MACD         : {'상승 모멘텀 ▲' if macd_bull else '하락 모멘텀 ▼'}")
    if target:
        print(f"  월가 목표가  : ${target:.2f}  ({upside_pct:+.1f}%)  "
              f"|  범위 ${t_low:.2f} ~ ${t_high:.2f}")

    # 리스크/보상
    print(f"\n{dash}")
    print(f"  📊 리스크 / 보상 분석")
    print(f"{dash}")
    if rr:
        print(f"  R/R 비율     : {rr}:1  →  {rr_label}")
        print(f"  기대 수익    : +{upside_pct}%  (월가 목표가 기준)")
        print(f"  최대 손실    : −{risk_pct}%  (ATR×2 손절 기준)")
    if hk_pct is not None:
        k_label = '적정' if 0 < hk_pct <= 25 else '과도 — 줄이는 것 권장' if hk_pct > 25 else '매수 비추천 (기대값 음수)'
        print(f"  하프켈리 비중: {hk_pct}%  (${hk_amt:,.0f})  →  {k_label}")
        print(f"  ※ 하프켈리 = 수학적 최적 베팅의 절반 (과집중 방지)")

    # 진입 전략 1 — 즉시 전액
    print(f"\n{dash}")
    print(f"  📥 진입 전략 1 — 즉시 전액 매수")
    print(f"{dash}")
    print(f"  매수가       : ${e1}  →  {sh1}주  (실투자금 ${cost_imm:,.2f})")
    if stop_loss:
        print(f"  손절가       : ${stop_loss}  (−{risk_pct}%,  현재가 − ATR×2)")
    if ml_imm:
        print(f"  최대 손실액  : −${ml_imm:,.2f}  (투자금의 −{round(ml_imm/capital*100,1)}%)")
    print(f"  적합 상황    : 상승 모멘텀 강하고 즉시 진입 확신 있을 때")

    # 진입 전략 2 — 분할매수 (권장)
    print(f"\n{dash}")
    print(f"  📥 진입 전략 2 — 3회 분할매수  (권장)")
    print(f"{dash}")
    print(f"  1차 (즉시, 33%)  : ${e1}  →  {sh1}주  (${a1:,.0f} 사용)")
    print(f"  2차 (−5% 하락)   : ${e2}  →  {sh2}주  (${a2:,.0f} 사용)")
    print(f"  3차 (−10% 또는 RSI<30): ${e3}  →  {sh3}주  (${a3:,.0f} 사용)")
    print(f"  ─ 평균 매수단가  : ${avg_cost}")
    print(f"  ─ 총 예상 주수   : {sh1+sh2+sh3}주")
    if stop_loss:
        print(f"  ─ 손절가         : ${stop_loss}  (1차 매수가 기준 ATR×2)")
    if ml_dca:
        print(f"  ─ 최대 손실액    : −${ml_dca:,.2f}  (투자금의 −{round(ml_dca/capital*100,1)}%)")
    print(f"  적합 상황    : 변동성 높거나 확신 부족할 때 — 평균 단가 낮추기 가능")

    # 익절 전략
    print(f"\n{dash}")
    print(f"  🎯 익절 전략")
    print(f"{dash}")
    print(f"  1차 익절 (+10%)  : ${tp1}  →  보유주의 30% 매도  (수익 일부 확보)")
    print(f"  2차 익절 (+20%)  : ${tp2}  →  보유주의 30% 추가 매도")
    if target:
        print(f"  3차 익절 (목표가): ${tp3}  →  나머지 전량 매도  (월가 컨센서스)")
    print(f"  추적 손절        : 신고점 갱신 후 ATR×1.5 하락 시 전량 매도")
    print(f"                   ※ 현재 기준 추적손절 초기값: ${trail_stop}")

    # 손절 규칙
    print(f"\n{dash}")
    print(f"  🛑 손절 & 재진입 규칙")
    print(f"{dash}")
    if stop_loss:
        print(f"  손절 트리거  : 종가 기준 ${stop_loss} 하향 이탈 시 즉시 전량 매도")
        print(f"               (장중 일시 이탈은 무시, 종가 확정 후 판단)")
    print(f"  재진입 조건  : 손절 후 최소 2주 대기 → quant 재실행 후 여전히")
    print(f"               상위권이면 재매수 가능 (단, 2차 이상 분할 가격 사용)")
    print(f"  퇴출 조건    : quant 재실행 후 순위권 이탈 시 익절 여부와 무관하게 매도")

    # 리밸런싱
    print(f"\n{dash}")
    print(f"  🔄 모니터링 & 리밸런싱")
    print(f"{dash}")
    print(f"  다음 점검일  : {rebal}  (3개월 후)")
    print(f"  점검 항목    : ① quant 재실행 → 순위 확인")
    print(f"               ② 재무 발표 후 EPS·매출 성장률 변화 확인")
    print(f"               ③ 섹터 리스크 이슈 없는지 뉴스 확인")
    print(f"  매도 신호    : 순위 이탈 / 손절선 이탈 / 펀더멘털 훼손")

    print(f"\n{sep}")
    print(f"  ⚠️  본 플랜은 참고용이며 투자 권유가 아닙니다.")
    print(f"     데이터 출처: yfinance  |  분석 기준일: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"{sep}\n")


# ── 실행 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  💼  개인 매매 플랜 생성기")
    print("  quant 스크리닝으로 발굴한 종목의 구체적 매매 전략을 제시합니다.")
    print("=" * 65)

    ticker_input = input("\n🔍 분석할 종목 티커 입력 (예: AAPL, NVDA, GILD): ").upper().strip()
    if not ticker_input:
        print("⚠️  티커를 입력해주세요.")
        exit()

    capital_raw = input("💵 투자 예정 금액 (USD, 예: 10000): $").strip().replace(',', '')
    try:
        capital_input = float(capital_raw)
        if capital_input <= 0:
            raise ValueError
    except ValueError:
        print("⚠️  올바른 금액을 입력해주세요 (숫자만).")
        exit()

    analyze_portfolio(ticker_input, capital_input)
