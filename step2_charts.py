import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MA_COLORS = {50: '#3498db', 100: '#e67e22', 150: '#2ecc71', 200: '#e74c3c'}


# ── 지표 계산 ───────────────────────────────────────────────

def _macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=sig, adjust=False).mean()
    return m, s, m - s


def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


# ── 서브플롯 그리기 ─────────────────────────────────────────

def _price(ax, df, spx_slice, ticker, label, use_candle):
    close = df['Close']

    if use_candle:
        up = df[df['Close'] >= df['Open']]
        dn = df[df['Close'] <  df['Open']]
        w  = 0.6  # matplotlib date unit = 1 day
        ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'],
               color='#2ecc71', width=w, alpha=0.85)
        ax.bar(dn.index, dn['Open'] - dn['Close'], bottom=dn['Close'],
               color='#e74c3c', width=w, alpha=0.85)
        ax.vlines(up.index, up['Low'], up['High'], color='#27ae60', linewidth=0.7)
        ax.vlines(dn.index, dn['Low'], dn['High'], color='#c0392b', linewidth=0.7)
        ax.set_title(f'[{ticker}] {label}  ·  캔들차트 + 이동평균선',
                     fontsize=11, fontweight='bold', pad=6)
    else:
        ax.plot(df.index, close, color='#2c3e50', alpha=0.2, linewidth=0.8, label='종가')
        if spx_slice is not None and len(spx_slice) > 5:
            spx_a = spx_slice.reindex(df.index, method='ffill').dropna()
            if len(spx_a) > 5:
                norm = spx_a / spx_a.iloc[0] * close.iloc[close.index.get_loc(spx_a.index[0])]
                ax.plot(norm.index, norm, color='#8e44ad', linewidth=1.2,
                        linestyle='--', label='S&P500 (정규화)', alpha=0.8)
        ax.set_title(f'[{ticker}] {label}  ·  이동평균선 + S&P500 비교',
                     fontsize=11, fontweight='bold', pad=6)

    for w_ma, c in MA_COLORS.items():
        if len(close) >= w_ma:
            ma = close.rolling(w_ma).mean()
            ax.plot(df.index, ma, color=c, linewidth=1.4,
                    label=f'{w_ma}MA', alpha=0.85)

    ax.legend(fontsize=8, loc='upper left', ncol=4)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('가격 (USD)', fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)


def _volume(ax, df):
    if 'Volume' not in df.columns or df['Volume'].isna().all():
        ax.text(0.5, 0.5, '거래량 데이터 없음', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return
    vol   = df['Volume']
    up_m  = df['Close'] >= df['Open']
    colors = ['#2ecc71' if m else '#e74c3c' for m in up_m]
    ax.bar(df.index, vol, color=colors, alpha=0.45, label='거래량')
    ax.plot(df.index, vol.rolling(20).mean(), color='#2c3e50',
            linewidth=1.1, label='거래량 20MA')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('거래량', fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x:.0f}'))
    plt.setp(ax.get_xticklabels(), visible=False)


def _macd_plot(ax, df):
    close        = df['Close']
    macd, sig, hist = _macd(close)
    h_colors     = ['#27ae60' if v >= 0 else '#e74c3c' for v in hist]
    ax.bar(df.index, hist, color=h_colors, alpha=0.55, label='히스토그램(다이버전스)')
    ax.plot(df.index, macd, color='#3498db', linewidth=1.1, label='MACD')
    ax.plot(df.index, sig,  color='#e74c3c', linewidth=1.0,
            linestyle='--', label='Signal')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=7, loc='upper left', ncol=3)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('MACD', fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)


def _slope_projection(ax, df, n_slope=10, n_future=90):
    """
    각 MA 끝점에서 선형 기울기(미분)를 계산하고 점선으로 연장.
    연장선끼리 교차하는 지점에 달러 가격과 날짜를 표시.
    n_slope  : 기울기 계산에 쓸 최근 거래일 수
    n_future : 미래로 연장할 거래일 수
    """
    close      = df['Close']
    last_date  = df.index[-1]
    future_idx = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=n_future)

    projections = {}

    for w, color in MA_COLORS.items():
        if len(close) < w + n_slope:
            continue
        ma = close.rolling(w).mean().dropna()
        if len(ma) < n_slope:
            continue

        # 최근 n_slope 포인트로 선형 회귀 → 기울기(1일당 가격 변화)
        y = ma.iloc[-n_slope:].values
        x = np.arange(n_slope)
        slope = np.polyfit(x, y, 1)[0]

        last_val    = float(ma.iloc[-1])
        future_vals = last_val + slope * np.arange(1, n_future + 1)

        # 점선으로 연장선 그리기
        ax.plot(
            [last_date] + list(future_idx),
            [last_val]  + list(future_vals),
            color=color, linewidth=1.1, linestyle=':', alpha=0.9,
        )

        projections[w] = dict(
            slope=slope, last_val=last_val,
            future_dates=future_idx, future_vals=future_vals,
        )

    # 교점 탐색 (모든 MA 쌍)
    keys     = sorted(projections.keys())
    labeled  = set()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            w1, w2 = keys[i], keys[j]
            p1, p2 = projections[w1], projections[w2]

            # y1 = a + s1*t ,  y2 = b + s2*t
            # 교점: t = (b - a) / (s1 - s2)
            s_diff = p1['slope'] - p2['slope']
            if abs(s_diff) < 1e-9:
                continue  # 평행 — 교점 없음

            t = (p2['last_val'] - p1['last_val']) / s_diff

            # 미래 창(0 < t ≤ n_future) 안에 있을 때만 표시
            if not (0 < t <= n_future):
                continue

            t_idx       = min(int(round(t)), len(p1['future_dates']) - 1)
            cross_price = p1['last_val'] + p1['slope'] * t
            cross_date  = p1['future_dates'][t_idx]

            # 중복 표시 방지 (가격 $5 이내 = 같은 교점으로 간주)
            dup = False
            for lp in labeled:
                if abs(lp - cross_price) < 5:
                    dup = True
                    break
            if dup:
                continue
            labeled.add(cross_price)

            ax.scatter([cross_date], [cross_price],
                       color='#2c3e50', s=65, zorder=12)
            ax.annotate(
                f"${cross_price:.1f}\n({cross_date.strftime('%m/%d')})",
                xy=(cross_date, cross_price),
                xytext=(12, 8), textcoords='offset points',
                fontsize=8, fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f1c40f',
                          alpha=0.92, edgecolor='#2c3e50', linewidth=0.8),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=0.9),
            )

    # x축 오른쪽 여백 확보 (연장선이 잘리지 않도록)
    if future_idx is not None and len(future_idx):
        ax.set_xlim(right=future_idx[-1] + pd.Timedelta(days=5))


def _rsi_plot(ax, df):
    close = df['Close']
    rsi   = _rsi(close)
    ax.plot(df.index, rsi, color='#9b59b6', linewidth=1.1, label='RSI(14)')
    ax.axhline(70, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axhline(30, color='#27ae60', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axhline(50, color='gray',    linewidth=0.4, alpha=0.5)
    ax.fill_between(df.index, 70, rsi.clip(upper=100),
                    where=(rsi > 70), color='#e74c3c', alpha=0.13, interpolate=True)
    ax.fill_between(df.index, rsi.clip(lower=0), 30,
                    where=(rsi < 30), color='#27ae60', alpha=0.13, interpolate=True)
    ax.set_ylim(0, 100)
    ax.set_ylabel('RSI', fontsize=8)
    ax.grid(True, alpha=0.2)

    last = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 50
    if last > 70:
        status, col = '과매수 ⚠️', '#e74c3c'
    elif last < 30:
        status, col = '과매도 💡', '#27ae60'
    else:
        status, col = '중립', 'gray'
    ax.text(0.99, 0.82, f'RSI {last:.1f}  ({status})',
            transform=ax.transAxes, fontsize=8, ha='right',
            color=col, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')


# ── 기간별 차트 1개 ──────────────────────────────────────────

def _draw(ticker, df, spx_slice, label, use_candle):
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'📈  [{ticker}]  {label}', fontsize=13, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[4, 1.2, 1.2, 1.0], hspace=0.06)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    _price(ax0, df, spx_slice, ticker, label, use_candle)
    _slope_projection(ax0, df)          # MA 기울기 연장선 + 교점 달러 표시
    _volume(ax1, df)
    _macd_plot(ax2, df)
    _rsi_plot(ax3, df)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


# ── 메인 함수 ────────────────────────────────────────────────

def analyze_charts(ticker, history_5y, spx_5y=None):
    """
    history_5y : stock.history(period='5y') 결과 (main에서 1회 호출)
    spx_5y     : ^GSPC history(period='5y') 결과 (main에서 1회 호출)
    """
    if history_5y is None or history_5y.empty:
        print(f'❌ [{ticker}] 주가 데이터 없음')
        return

    # timezone 제거
    if history_5y.index.tz is not None:
        history_5y = history_5y.copy()
        history_5y.index = history_5y.index.tz_convert(None)

    spx_close = None
    if spx_5y is not None and not spx_5y.empty:
        spx_5y = spx_5y.copy()
        if spx_5y.index.tz is not None:
            spx_5y.index = spx_5y.index.tz_convert(None)
        spx_close = spx_5y['Close']

    now = history_5y.index[-1]

    periods = [
        ('1년 — 단기·중기 트렌드',
         history_5y[history_5y.index >= now - pd.DateOffset(years=1)],
         True),   # 캔들차트
        ('3년 — 중장기 퍼포먼스',
         history_5y[history_5y.index >= now - pd.DateOffset(years=3)],
         False),
        ('5년 — 장기 퍼포먼스',
         history_5y,
         False),
    ]

    print(f'\n📈 STEP 2 — [{ticker}] 기술적 차트 분석 (3개 기간)\n')

    for label, df, use_candle in periods:
        if df.empty:
            print(f'  ⚠️ {label}: 데이터 부족 (건너뜀)')
            continue
        spx_slice = (spx_close[spx_close.index >= df.index[0]]
                     if spx_close is not None else None)
        _draw(ticker, df, spx_slice, label, use_candle)
        print(f'  ✅ {label} 출력 완료')
