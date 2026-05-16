import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore', category=UserWarning)   # 한글 폰트 등 matplotlib UserWarning
warnings.filterwarnings('ignore', category=FutureWarning)  # pandas FutureWarning

from IPython.display import display, HTML

try:
    import ipywidgets as widgets
    from IPython.display import display as ipy_display
    _WIDGETS_OK = True
except ImportError:
    _WIDGETS_OK = False

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MA_COLORS = {50: '#3498db', 100: '#e67e22', 150: '#2ecc71', 200: '#e74c3c'}


# ── 지표 계산 ────────────────────────────────────────────────────────────────

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


# ── 서브플롯 그리기 ──────────────────────────────────────────────────────────

def _bollinger(ax, close, window=20, num_std=2):
    """볼린저밴드 (MA20 ± 2σ) — 음영 + 경계선"""
    if len(close) < window:
        return
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    ax.fill_between(close.index, lower, upper,
                    alpha=0.08, color='#95a5a6', label=f'BB({window},{num_std}σ)')
    ax.plot(close.index, upper, color='#95a5a6', linewidth=0.7,
            linestyle='--', alpha=0.65)
    ax.plot(close.index, lower, color='#95a5a6', linewidth=0.7,
            linestyle='--', alpha=0.65)


def _price(ax, df, spx_slice, ticker, label, use_candle):
    close = df['Close']
    _bollinger(ax, close)          # 볼린저밴드 — MA선 아래에 먼저 그림

    if use_candle:
        up = df[df['Close'] >= df['Open']]
        dn = df[df['Close'] <  df['Open']]
        w  = 0.6
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


def _obv_line(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _volume(ax, df):
    if 'Volume' not in df.columns or df['Volume'].isna().all():
        ax.text(0.5, 0.5, '거래량 데이터 없음', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return
    vol    = df['Volume']
    up_m   = df['Close'] >= df['Open']
    colors = ['#2ecc71' if m else '#e74c3c' for m in up_m]
    ax.bar(df.index, vol, color=colors, alpha=0.45, label='거래량')
    ax.plot(df.index, vol.rolling(20).mean(), color='#2c3e50',
            linewidth=1.1, label='거래량 20MA')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('거래량', fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x:.0f}'))

    # OBV (기관 누적매수/매도 신호) — 우측 축
    ax_obv = ax.twinx()
    obv = _obv_line(df['Close'], df['Volume'])
    ax_obv.plot(df.index, obv, color='#9b59b6', linewidth=1.0, alpha=0.75, label='OBV')
    ax_obv.set_ylabel('OBV', fontsize=7, color='#9b59b6')
    ax_obv.tick_params(axis='y', labelcolor='#9b59b6', labelsize=7)
    ax_obv.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M' if abs(x) >= 1e6 else f'{x:.0f}'))
    ax_obv.legend(fontsize=7, loc='upper right')

    plt.setp(ax.get_xticklabels(), visible=False)


def _macd_plot(ax, df):
    close          = df['Close']
    macd, sig, hist = _macd(close)
    h_colors       = ['#27ae60' if v >= 0 else '#e74c3c' for v in hist]
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

        y = ma.iloc[-n_slope:].values
        x = np.arange(n_slope)
        slope = np.polyfit(x, y, 1)[0]

        last_val    = float(ma.iloc[-1])
        future_vals = last_val + slope * np.arange(1, n_future + 1)

        ax.plot(
            [last_date] + list(future_idx),
            [last_val]  + list(future_vals),
            color=color, linewidth=1.1, linestyle=':', alpha=0.9,
        )

        projections[w] = dict(
            slope=slope, last_val=last_val,
            future_dates=future_idx, future_vals=future_vals,
        )

    keys    = sorted(projections.keys())
    labeled = set()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            w1, w2 = keys[i], keys[j]
            p1, p2 = projections[w1], projections[w2]

            s_diff = p1['slope'] - p2['slope']
            if abs(s_diff) < 1e-9:
                continue

            t = (p2['last_val'] - p1['last_val']) / s_diff

            if not (0 < t <= n_future):
                continue

            t_idx       = min(int(round(t)), len(p1['future_dates']) - 1)
            cross_price = p1['last_val'] + p1['slope'] * t
            cross_date  = p1['future_dates'][t_idx]

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


# ── 기간별 차트 1개 ──────────────────────────────────────────────────────────

def _draw(ticker, df, spx_slice, label, use_candle):
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'📈  [{ticker}]  {label}', fontsize=13, fontweight='bold', y=0.995)

    gs  = gridspec.GridSpec(4, 1, figure=fig,
                            height_ratios=[4, 1.2, 1.2, 1.0], hspace=0.06)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    _price(ax0, df, spx_slice, ticker, label, use_candle)
    _slope_projection(ax0, df)
    _volume(ax1, df)
    _macd_plot(ax2, df)
    _rsi_plot(ax3, df)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


# ── 기울기 & 교점 텍스트 요약 ────────────────────────────────────────────────

def _slope_summary_html(df, ticker, spx_close=None, n_slope=10):
    """이평선 기울기 방향 + 최근 교점 + 기술적 종합점수 + RS 분석 HTML 반환"""
    close = df['Close']

    # 기울기 계산
    slopes  = {}
    ma_vals = {}
    for w in MA_COLORS:
        if len(close) < w + n_slope:
            continue
        ma = close.rolling(w).mean().dropna()
        if len(ma) < n_slope:
            continue
        y = ma.iloc[-n_slope:].values
        slopes[w]  = np.polyfit(np.arange(n_slope), y, 1)[0]
        ma_vals[w] = float(ma.iloc[-1])

    # 최근 교점 탐색 (50MA 기준)
    cross_points = []
    ma50 = close.rolling(50).mean()
    for other in [100, 150, 200]:
        if other not in ma_vals:
            continue
        ma_other  = close.rolling(other).mean()
        diff      = ma50 - ma_other
        crossings = df[(diff.shift(1) * diff < 0) & diff.notna() & diff.shift(1).notna()]
        if crossings.empty:
            continue
        last      = crossings.iloc[-1]
        price     = (ma50.loc[last.name] + ma_other.loc[last.name]) / 2
        is_golden = diff.loc[last.name] > 0
        cross_points.append({
            'pair':  f'50MA × {other}MA',
            'date':  last.name.strftime('%Y-%m-%d'),
            'price': price,
            'type':  '골든크로스 ↑' if is_golden else '데드크로스 ↓',
            'color': '#27ae60'       if is_golden else '#e74c3c',
        })

    # 기울기 테이블 행
    slope_rows = ''
    for w, color in MA_COLORS.items():
        if w not in slopes:
            continue
        s   = slopes[w]
        val = ma_vals[w]
        dir_label = '↗ 상승세' if s > 0 else '↘ 하락세'
        dir_color = '#27ae60'  if s > 0 else '#e74c3c'
        slope_rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
          <td style="padding:6px 12px">
            <span style="background:{color};color:white;border-radius:4px;
                         padding:2px 8px;font-size:12px;font-weight:bold">{w}MA</span>
          </td>
          <td style="padding:6px 12px;font-size:13px;font-weight:bold">${val:.2f}</td>
          <td style="padding:6px 12px;font-size:12px;font-family:monospace;color:#555">{s:+.4f}</td>
          <td style="padding:6px 12px;font-size:13px;font-weight:bold;color:{dir_color}">{dir_label}</td>
        </tr>'''

    # 교점 테이블 행
    cross_rows = ''
    for cp in cross_points:
        cross_rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
          <td style="padding:6px 12px;font-size:12px;font-weight:bold;color:#2c3e50">{cp["pair"]}</td>
          <td style="padding:6px 12px;font-size:13px;font-weight:bold">${cp["price"]:.2f}</td>
          <td style="padding:6px 12px;font-size:12px;color:#7f8c8d">{cp["date"]}</td>
          <td style="padding:6px 12px">
            <span style="background:{cp["color"]};color:white;font-size:11px;
                         padding:2px 9px;border-radius:10px">{cp["type"]}</span>
          </td>
        </tr>'''
    if not cross_rows:
        cross_rows = '<tr><td colspan="4" style="padding:8px;color:#95a5a6;font-size:12px;text-align:center">해당 기간 내 교점 없음</td></tr>'

    # ── 기술적 종합점수 (0~100) ──────────────────────────────────────────────
    cur_price = float(close.iloc[-1])
    tech_score = 0
    # MA 정배열 여부 (35점): 50>150>200=완전정배열, 50>200=부분
    if all(w in ma_vals for w in [50, 200]):
        if ma_vals[50] > ma_vals.get(150, 0) and ma_vals.get(150, 0) > ma_vals[200]:
            tech_score += 35
        elif ma_vals[50] > ma_vals[200]:
            tech_score += 25
        elif ma_vals[50] > ma_vals.get(150, 0):
            tech_score += 15
    # 현재가 vs 200MA (25점)
    if 200 in ma_vals and cur_price > ma_vals[200]:
        tech_score += 25
    # MACD 크로스 (20점): MACD > Signal
    try:
        macd_line, sig_line, _ = _macd(close)
        if float(macd_line.iloc[-1]) > float(sig_line.iloc[-1]):
            tech_score += 20
    except Exception:
        pass
    # RSI 40~65 최적구간 (20점)
    try:
        rsi_val = float(_rsi(close).iloc[-1])
        if 40 <= rsi_val <= 65:
            tech_score += 20
        elif 30 <= rsi_val < 40 or 65 < rsi_val <= 70:
            tech_score += 10
    except Exception:
        pass
    ts_col = '#27ae60' if tech_score >= 70 else '#f39c12' if tech_score >= 45 else '#e74c3c'
    ts_lbl = '강세' if tech_score >= 70 else '중립' if tech_score >= 45 else '약세'

    tech_score_html = f'''
    <div style="margin-top:12px;background:#fff;border:1px solid #ecf0f1;
                border-radius:6px;padding:10px 14px;display:flex;align-items:center;gap:18px">
      <span style="font-size:12px;font-weight:bold;color:#7f8c8d;white-space:nowrap">▸ 기술적 종합점수</span>
      <span style="font-size:22px;font-weight:bold;color:{ts_col}">{tech_score} / 100</span>
      <span style="background:{ts_col};color:white;border-radius:5px;
                   padding:3px 12px;font-size:12px;font-weight:bold">{ts_lbl}</span>
      <span style="font-size:11px;color:#95a5a6">MA정배열(35) + 200MA위(25) + MACD(20) + RSI(20)</span>
    </div>'''

    # ── S&P500 상대강도 (RS) ─────────────────────────────────────────────────
    rs_html = ''
    if spx_close is not None:
        try:
            spx_al = spx_close.reindex(close.index, method='ffill').dropna()
            common = close.index.intersection(spx_al.index)
            if len(common) > 63:
                n = min(252, len(common))
                stk_ret = (close.loc[common[-1]] / close.loc[common[-n]] - 1) * 100
                spx_ret = (spx_al.loc[common[-1]] / spx_al.loc[common[-n]] - 1) * 100
                rs_diff = round(stk_ret - spx_ret, 1)
                period_lbl = '12개월' if n >= 252 else f'{n}거래일'
                rs_col = '#27ae60' if rs_diff >= 0 else '#e74c3c'
                rs_txt = f'S&P500 아웃퍼폼 ▲ ({period_lbl})' if rs_diff >= 0 else f'S&P500 언더퍼폼 ▼ ({period_lbl})'
                rs_html = f'''
    <div style="margin-top:8px;background:#fff;border:1px solid #ecf0f1;
                border-radius:6px;padding:10px 14px">
      <span style="font-size:12px;font-weight:bold;color:#7f8c8d">▸ S&P500 상대강도 (RS · {period_lbl})</span>
      <div style="margin-top:6px;font-size:13px">
        종목 수익률: <b style="color:#2c3e50">{stk_ret:+.1f}%</b> &nbsp;|&nbsp;
        S&P500: <b style="color:#8e44ad">{spx_ret:+.1f}%</b> &nbsp;|&nbsp;
        <span style="color:{rs_col};font-weight:bold">RS 차이: {rs_diff:+.1f}%</span>
        &nbsp;—&nbsp; <span style="color:{rs_col}">{rs_txt}</span>
      </div>
    </div>'''
        except Exception:
            pass

    return f'''
    <div style="font-family:Arial,sans-serif;max-width:940px;margin:10px 0 22px;
                background:#f8f9fa;border-radius:8px;padding:16px;
                border:1px solid #dde4e9">
      <h4 style="margin:0 0 12px;color:#2c3e50;font-size:14px">
        📐 [{ticker}] 이평선 기울기 & 교점 분석 (1년 기준)
      </h4>
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <div style="flex:1;min-width:290px">
          <p style="margin:0 0 5px;font-size:12px;font-weight:bold;color:#7f8c8d">▸ 기울기 방향</p>
          <table style="border-collapse:collapse;width:100%;background:#fff;
                        border:1px solid #ecf0f1;border-radius:6px;overflow:hidden">
            <thead>
              <tr style="background:#2c3e50;color:white;font-size:12px">
                <th style="padding:6px 12px;text-align:left">이평선</th>
                <th style="padding:6px 12px">현재값</th>
                <th style="padding:6px 12px">기울기</th>
                <th style="padding:6px 12px">방향</th>
              </tr>
            </thead>
            <tbody>{slope_rows}</tbody>
          </table>
        </div>
        <div style="flex:1;min-width:290px">
          <p style="margin:0 0 5px;font-size:12px;font-weight:bold;color:#7f8c8d">▸ 최근 50MA 교점</p>
          <table style="border-collapse:collapse;width:100%;background:#fff;
                        border:1px solid #ecf0f1;border-radius:6px;overflow:hidden">
            <thead>
              <tr style="background:#2c3e50;color:white;font-size:12px">
                <th style="padding:6px 12px;text-align:left">교점</th>
                <th style="padding:6px 12px">가격</th>
                <th style="padding:6px 12px">날짜</th>
                <th style="padding:6px 12px">종류</th>
              </tr>
            </thead>
            <tbody>{cross_rows}</tbody>
          </table>
        </div>
      </div>
      {tech_score_html}
      {rs_html}
      <p style="font-size:11px;color:#95a5a6;margin:8px 0 0">
        ※ 기울기: 최근 {n_slope}거래일 선형회귀 (1일당 달러 변화) &nbsp;|&nbsp;
        골든크로스: 50MA가 상위선 상향 돌파 &nbsp;|&nbsp; 데드크로스: 50MA가 하향 이탈
      </p>
    </div>'''


# ── 토글 차트 (3년/5년용) ────────────────────────────────────────────────────

def _toggle_chart(ticker, df, spx_close, label, use_candle):
    """ipywidgets 버튼으로 펼치기/접기 가능한 차트"""
    if not _WIDGETS_OK:
        # ipywidgets 없으면 그냥 출력
        spx_slice = (spx_close[spx_close.index >= df.index[0]]
                     if spx_close is not None else None)
        _draw(ticker, df, spx_slice, label, use_candle)
        return

    btn = widgets.ToggleButton(
        value=False,
        description=f'📊 {label} 보기',
        button_style='info',
        layout=widgets.Layout(width='380px', height='38px'),
    )
    out      = widgets.Output()
    rendered = [False]

    def on_toggle(change):
        if change['new']:
            btn.description = f'🔼 {label} 숨기기'
            if not rendered[0]:
                spx_slice = (spx_close[spx_close.index >= df.index[0]]
                             if spx_close is not None else None)
                with out:
                    _draw(ticker, df, spx_slice, label, use_candle)
                rendered[0] = True
            out.layout.display = ''
        else:
            btn.description = f'📊 {label} 보기'
            out.layout.display = 'none'

    out.layout.display = 'none'
    btn.observe(on_toggle, names='value')
    ipy_display(btn, out)


# ── 메인 함수 ────────────────────────────────────────────────────────────────

def analyze_charts(ticker, history_5y, spx_5y=None):
    """
    history_5y : stock.history(period='5y') 결과 (main에서 1회 호출)
    spx_5y     : ^GSPC history(period='5y') 결과 (main에서 1회 호출)
    """
    if history_5y is None or history_5y.empty:
        print(f'❌ [{ticker}] 주가 데이터 없음')
        return

    if history_5y.index.tz is not None:
        history_5y = history_5y.copy()
        history_5y.index = history_5y.index.tz_convert(None)

    spx_close = None
    if spx_5y is not None and not spx_5y.empty:
        spx_5y = spx_5y.copy()
        if spx_5y.index.tz is not None:
            spx_5y.index = spx_5y.index.tz_convert(None)
        spx_close = spx_5y['Close']

    now   = history_5y.index[-1]
    df_1y = history_5y[history_5y.index >= now - pd.DateOffset(years=1)]
    df_3y = history_5y[history_5y.index >= now - pd.DateOffset(years=3)]
    df_5y = history_5y

    print(f'\n📈 STEP 2 — [{ticker}] 기술적 차트 분석\n')

    # 1년 차트 바로 출력
    if not df_1y.empty:
        spx_1y = (spx_close[spx_close.index >= df_1y.index[0]]
                  if spx_close is not None else None)
        _draw(ticker, df_1y, spx_1y, '1년 — 단기·중기 트렌드', True)
        display(HTML(_slope_summary_html(df_1y, ticker, spx_close=spx_close)))
        print('  ✅ 1년 차트 출력 완료')
    else:
        print('  ⚠️ 1년: 데이터 부족')

    # 3년 / 5년 차트 — 토글 버튼
    print('\n  📁 장기 차트 (버튼을 눌러 펼치세요):')
    for label, df in [('3년 — 중장기 퍼포먼스', df_3y), ('5년 — 장기 퍼포먼스', df_5y)]:
        if df.empty:
            print(f'  ⚠️ {label}: 데이터 부족 (건너뜀)')
            continue
        _toggle_chart(ticker, df, spx_close, label, False)
