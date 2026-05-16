import pandas as pd
import numpy as np
import yfinance as yf
from IPython.display import display, HTML


# ── 상수 ─────────────────────────────────────────────────────
FED_RATE_RANGE = '4.25% ~ 4.50%'  # FOMC 결정 기준금리 (변경 시 FED_RATE_DATE도 함께 업데이트)
FED_RATE_DATE  = '2025-01-29'      # 위 상수의 기준일 — FRED 조회 실패 시 대체값으로 사용
_RATE_CACHE: dict = {}             # 세션 내 금리 캐시 (1시간 유지)


# ══════════════════════════════════════════════════════════════
# 거시경제 캘린더 (하드코딩 — 2025~2026)
# ※ 날짜는 공식 발표 기준이나 ±1~2일 오차 가능. 투자 전 직접 확인 권장.
# ══════════════════════════════════════════════════════════════
MACRO_CALENDAR = [
    # (날짜, 이벤트명, 출처, 카테고리, 아이콘)
    # ── FOMC 2025 ──
    ('2025-06-18', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2025-07-30', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2025-09-17', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2025-10-29', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2025-12-17', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    # ── CPI 2025 ──
    ('2025-06-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-07-15', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-08-12', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-09-12', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-10-15', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-11-13', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2025-12-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    # ── NFP 2025 ──
    ('2025-06-06', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-07-03', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-08-01', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-09-05', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-10-03', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-11-07', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2025-12-05', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    # ── FOMC 2026 (추정) ──
    ('2026-01-28', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-03-18', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-04-29', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-06-10', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-07-29', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-09-16', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-10-28', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    ('2026-12-16', 'FOMC 금리 결정',  '연준(Fed)', '금리',       '🏦'),
    # ── CPI 2026 (추정) ──
    ('2026-01-14', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-02-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-03-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-04-10', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-05-13', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-06-10', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-07-14', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-08-12', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-09-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-10-14', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-11-12', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    ('2026-12-11', 'CPI 소비자물가',  'BLS',       '인플레이션', '📊'),
    # ── NFP 2026 (추정) ──
    ('2026-01-09', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-02-06', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-03-06', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-04-03', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-05-08', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-06-05', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-07-10', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-08-07', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-09-04', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-10-02', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-11-06', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
    ('2026-12-04', '고용보고서(NFP)', 'BLS',       '고용',       '👷'),
]

# ══════════════════════════════════════════════════════════════
# 섹터별 리스크 요인 (yfinance sector 명칭 기준)
# ══════════════════════════════════════════════════════════════
SECTOR_RISKS = {
    'Technology': [
        ('금리 인상',        '성장주 DCF 할인율 상승 → 고PER 밸류에이션 직격'),
        ('반독점·AI 규제',   'EU·미국 빅테크 규제 강화 시 핵심 사업 모델 위협'),
        ('반도체 수출 통제', '대중 수출 제한 확대 시 매출 직접 감소'),
        ('달러 강세',        '해외 매출 비중 높은 경우 환율 역풍'),
    ],
    'Financial Services': [
        ('금리 급락',         '순이자마진(NIM) 축소 → 은행·보험 수익성 직결'),
        ('신용 스프레드 확대','경기 침체 시 대출 부실화·대손충당금 급증'),
        ('규제 자본 강화',    '바젤IV 자본 적정성 강화 시 배당·자사주 매입 제한'),
        ('핀테크 경쟁',       '디지털 뱅킹 성장으로 전통 금융 고객 이탈 가속'),
    ],
    'Healthcare': [
        ('FDA 승인 불확실성', '핵심 신약 임상 실패·승인 지연 시 주가 급락 위험'),
        ('특허 만료',         '블록버스터 의약품 특허 만료 → 제네릭 출시로 매출 절벽'),
        ('메디케어 약가 협상','정부 약가 인하 압력 확대 시 수익성 압박'),
        ('임상시험 실패',     '파이프라인 가치 급감 → 시가총액 10~50% 하락 가능'),
    ],
    'Energy': [
        ('국제유가 하락',    'WTI/Brent 하락 시 생산·탐사 수익성 급격히 악화'),
        ('OPEC+ 증산',       '공급 과잉으로 유가 압박 → 현금흐름 감소'),
        ('탄소세·환경 규제', 'ESG 강화·탄소세 부과 시 전통 에너지 설비투자 제한'),
        ('재생에너지 전환',  '장기 화석연료 수요 감소 구조적 리스크'),
    ],
    'Consumer Cyclical': [
        ('소비자 심리 악화', '경기 침체 시 재량 소비 급감 — 가장 민감한 섹터'),
        ('인플레이션·금리',  '원가 상승 + 소비자 대출 부담 증가 → 이중 압박'),
        ('공급망 차질',      '물류 비용 증가·재고 적체 → 마진 하락'),
        ('경쟁 심화',        'e커머스 가격 경쟁 격화로 오프라인 업체 점유율 잠식'),
    ],
    'Consumer Defensive': [
        ('원재료 인플레이션','식품·음료 원가 급등 시 마진 압박 (가격 전가 한계)'),
        ('경쟁 심화',        '저가 PB(자체 브랜드) 성장으로 브랜드 점유율 잠식'),
        ('성장 둔화',        '방어주 특성상 경기 호황기엔 성장주 대비 소외 가능'),
    ],
    'Real Estate': [
        ('금리 인상',          '모기지 금리 직결 → 부동산 거래량·가치 동반 하락'),
        ('상업용 부동산 위기', '재택근무 정착으로 오피스 공실률 구조적 상승'),
        ('자본 조달 비용',     '금리 상승 시 리츠 배당 매력 감소·레버리지 비용 증가'),
    ],
    'Industrials': [
        ('경기 침체(PMI 하락)', '제조업 지표 악화 → 설비 투자 급감'),
        ('무역 분쟁·관세',     '수출 의존 기업은 관세 부과 시 직격탄'),
        ('공급망 병목',        '핵심 부품 조달 차질 시 생산 차질·납기 지연'),
        ('에너지 비용 급등',   '에너지 집약 산업은 유가·전기료 상승 시 원가 압박'),
    ],
    'Communication Services': [
        ('광고 시장 침체', '경기 둔화 시 디지털 광고 집행 급감 — 매출 직결'),
        ('스트리밍 경쟁',  '구독자 이탈·ARPU 하락·콘텐츠 비용 증가'),
        ('플랫폼 규제',    '콘텐츠 책임 강화 법안 → 운영 비용 급증'),
        ('AI 대체 위험',   'AI 검색·생성 서비스 확산으로 전통 광고 모델 잠식'),
    ],
    'Utilities': [
        ('금리 인상',       '채권 대체재 성격 — 금리 상승 시 배당 매력 급감'),
        ('규제 요금 동결',  '전기·가스 요금 인상 제한 시 비용 전가 불가'),
        ('재생에너지 투자', '에너지 전환 설비투자 급증 → 부채 증가·배당 압박'),
    ],
    'Basic Materials': [
        ('중국 경기 둔화', '원자재 최대 소비국 수요 감소 → 가격 급락'),
        ('달러 강세',      '원자재는 달러 표시 — 달러 강세 시 가격 하락 압력'),
        ('공급 과잉',      '광산·화학 증설 경쟁으로 마진 구조적 압박'),
    ],
}

DEFAULT_RISKS = [
    ('금리 변동',       '연준 금리 정책 변화는 전 섹터에 영향'),
    ('달러 강세',       '해외 매출 비중 높은 기업은 환율 역풍'),
    ('지정학적 리스크', '무역 분쟁·전쟁·제재 등 예측 불가 이벤트'),
    ('경기 침체',       '전반적 소비·투자 위축 시 수익성 악화'),
]


# ══════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════

def _parse_news(raw_news):
    items = []
    for art in (raw_news or [])[:6]:
        try:
            content = art.get('content', {})
            if content and isinstance(content, dict):
                title  = content.get('title', '제목 없음') or '제목 없음'
                ts     = content.get('pubDate', '')
                date   = str(ts)[:10]
                prov   = content.get('provider', {})
                source = prov.get('displayName', '') if isinstance(prov, dict) else ''
                url_d  = content.get('canonicalUrl', {})
                link   = url_d.get('url', '') if isinstance(url_d, dict) else ''
            else:
                title  = art.get('title', '제목 없음') or '제목 없음'
                ts     = art.get('providerPublishTime', '')
                date   = (pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d')
                          if isinstance(ts, (int, float)) and ts else str(ts)[:10])
                source = art.get('publisher', '') or ''
                link   = art.get('link', '') or ''
            items.append({'title': title, 'date': date, 'source': source, 'link': link})
        except Exception:
            continue
    return items


def _parse_eps(raw_eps):
    """EPS history 파싱 → (next_date_str, [past_quarters])"""
    # ── Bug fix 1: raw_eps 타입 검증 ──────────────────────────
    if raw_eps is None:
        return None, []
    if not isinstance(raw_eps, pd.DataFrame):
        return None, []
    if raw_eps.empty:
        return None, []

    try:
        df = raw_eps.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 컬럼명 정규화 (yfinance 버전별 차이 대응)
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(' ', '').replace('_', '')
            if 'estimate' in cl:  col_map[c] = 'EPS Estimate'
            elif 'reported' in cl: col_map[c] = 'Reported EPS'
            elif 'surprise' in cl: col_map[c] = 'Surprise'
        df = df.rename(columns=col_map)

        # ── Bug fix 1: df.get() 대신 명시적 컬럼 존재 확인 ──────
        if 'Reported EPS' not in df.columns:
            return None, []

        past   = df[df['Reported EPS'].notna()].head(4)
        future = df[df['Reported EPS'].isna()]

        next_date = None
        if not future.empty:
            # 가장 가까운 미래 날짜 (인덱스 역순 중 마지막 = 가장 이른 미래)
            next_date = str(future.index[-1])[:10]

        history = []
        for date, row in past.iterrows():
            est  = row.get('EPS Estimate', np.nan)
            act  = row.get('Reported EPS', np.nan)
            surp = row.get('Surprise', np.nan)
            history.append({
                'date':     str(date)[:7],
                'estimate': f'${est:.2f}' if pd.notna(est)  else 'N/A',
                'actual':   f'${act:.2f}' if pd.notna(act)  else 'N/A',
                'surprise': surp if (pd.notna(surp) and isinstance(surp, (int, float))) else None,
            })
        return next_date, history

    except Exception:
        return None, []


def _ts_to_date(ts):
    """Unix 타임스탬프 → 날짜 문자열. 실패 시 'N/A'."""
    try:
        if ts is None or ts == 0:
            return 'N/A'
        return pd.Timestamp(int(ts), unit='s').strftime('%Y-%m-%d')
    except Exception:
        return 'N/A'


def _upcoming_macro(days_ahead=60):
    today  = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    events = []
    for row in MACRO_CALENDAR:
        d = pd.Timestamp(row[0])
        if today <= d <= cutoff:
            events.append({
                'date': d, 'name': row[1], 'source': row[2],
                'category': row[3], 'icon': row[4],
                'days': (d - today).days,
            })
    return sorted(events, key=lambda x: x['date'])


def _analyst_info(stock, info):
    data = {
        'earnings_date': 'N/A',
        'target_mean':   info.get('targetMeanPrice'),
        'target_high':   info.get('targetHighPrice'),
        'target_low':    info.get('targetLowPrice'),
        'n_analysts':    int(info.get('numberOfAnalystOpinions') or 0),
        'rec_key':       info.get('recommendationKey') or '',
    }
    try:
        cal = stock.calendar if stock else None
        # ── Bug fix 3: calendar가 dict인지 먼저 확인 ────────────
        if not isinstance(cal, dict):
            return data
        dates = cal.get('Earnings Date') or cal.get('earningsDate')
        if isinstance(dates, list) and dates:
            data['earnings_date'] = str(dates[0])[:10]
        elif dates is not None:
            data['earnings_date'] = str(dates)[:10]
    except Exception:
        pass
    return data


def _dividend_block(info, raw_dividends):
    rate   = info.get('dividendRate') or info.get('lastDividendValue')
    yld    = info.get('dividendYield')
    payout = info.get('payoutRatio')
    ex_ts  = info.get('exDividendDate')
    pay_ts = info.get('lastDividendDate')

    has_history = (isinstance(raw_dividends, pd.Series)
                   and not raw_dividends.empty)

    if not rate and not has_history:
        return '''
        <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
          <h3 style="margin:0 0 6px;font-size:14px;color:#2c3e50">💰 배당 정보</h3>
          <p style="margin:0;font-size:13px;color:#7f8c8d">이 종목은 배당금을 지급하지 않습니다 <b>(무배당주)</b></p>
        </div>'''

    if not rate and has_history:
        rate = float(raw_dividends.iloc[-1])

    rate_str   = f'${rate:.4f}'        if rate   else 'N/A'
    yld_str    = f'{yld*100:.2f}%'     if yld    else 'N/A'
    payout_str = f'{payout*100:.1f}%'  if payout else 'N/A'
    ex_date    = _ts_to_date(ex_ts)
    pay_date   = _ts_to_date(pay_ts)

    # 배당 이력 (최근 8회, 변화량 표시)
    div_rows = ''
    if has_history:
        recent = raw_dividends.tail(8)
        vals   = list(recent.items())
        for i, (date, amount) in enumerate(reversed(vals)):
            prev_amount = vals[len(vals) - 2 - i][1] if i < len(vals) - 1 else None
            if prev_amount is not None:
                arrow     = '▲' if amount > prev_amount else '▼' if amount < prev_amount else '─'
                arrow_col = ('#27ae60' if amount > prev_amount
                             else '#e74c3c' if amount < prev_amount else 'gray')
            else:
                arrow, arrow_col = '─', 'gray'
            div_rows += f'''
            <tr style="border-bottom:1px solid #ecf0f1">
              <td style="padding:5px 10px;font-size:12px">{str(date)[:10]}</td>
              <td style="padding:5px 10px;text-align:right;font-size:12px;font-weight:bold">${float(amount):.4f}</td>
              <td style="padding:5px 10px;text-align:center;color:{arrow_col};font-weight:bold">{arrow}</td>
            </tr>'''
    else:
        div_rows = '<tr><td colspan="3" style="padding:8px;color:#95a5a6;font-size:12px">배당 이력 없음</td></tr>'

    return f'''
    <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
      <h3 style="margin:0 0 10px;font-size:14px;color:#2c3e50">💰 배당 정보</h3>
      <table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:10px">
        <tr>
          <td style="padding:5px 14px;color:#555"><b>연간 배당금</b></td>
          <td style="padding:5px;font-weight:bold;color:#2980b9">{rate_str} / 주</td>
          <td style="padding:5px 14px;color:#555"><b>배당 수익률</b></td>
          <td style="padding:5px;font-weight:bold;color:#27ae60">{yld_str}</td>
        </tr>
        <tr>
          <td style="padding:5px 14px;color:#555"><b>배당락일 (Ex-Date)</b></td>
          <td style="padding:5px;font-weight:bold;color:#e74c3c">{ex_date}</td>
          <td style="padding:5px 14px;color:#555"><b>배당 성향 (Payout)</b></td>
          <td style="padding:5px">{payout_str}</td>
        </tr>
        <tr>
          <td style="padding:5px 14px;color:#555"><b>배당 지급일</b></td>
          <td style="padding:5px">{pay_date}</td>
          <td></td><td></td>
        </tr>
      </table>
      <details>
        <summary style="cursor:pointer;font-size:12px;color:#2980b9;font-weight:bold;margin-top:4px">
          최근 배당 이력 보기 (클릭)
        </summary>
        <table style="border-collapse:collapse;width:260px;margin-top:8px;
                      border:1px solid #ecf0f1;border-radius:4px;overflow:hidden">
          <thead>
            <tr style="background:#ecf0f1;font-size:12px">
              <th style="padding:5px 10px;text-align:left">지급일</th>
              <th style="padding:5px 10px">금액</th>
              <th style="padding:5px 10px">변화</th>
            </tr>
          </thead>
          <tbody>{div_rows}</tbody>
        </table>
      </details>
    </div>'''


def _macro_block(upcoming):
    category_colors = {
        '금리':       '#e74c3c',
        '인플레이션': '#e67e22',
        '고용':       '#3498db',
    }
    if not upcoming:
        last_cal = max(pd.Timestamp(r[0]) for r in MACRO_CALENDAR)
        if last_cal < pd.Timestamp.now().normalize():
            msg   = f'캘린더 만료 ({last_cal.strftime("%Y-%m-%d")} 이후 데이터 없음) — MACRO_CALENDAR 업데이트 필요'
            p_col = '#e74c3c'
        else:
            msg   = '향후 60일 이내 주요 일정 없음'
            p_col = '#95a5a6'
        return f'''
        <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
          <h3 style="margin:0 0 6px;font-size:14px;color:#2c3e50">🌍 향후 60일 거시경제 일정</h3>
          <p style="margin:0;font-size:12px;color:{p_col}">{msg}</p>
        </div>'''

    rows = ''
    for ev in upcoming:
        col   = category_colors.get(ev['category'], '#95a5a6')
        d_str = ev['date'].strftime('%Y-%m-%d (%a)')
        d_lbl = f"D+{ev['days']}" if ev['days'] > 0 else 'Today'
        rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
          <td style="padding:6px 10px;font-size:12px;font-weight:bold;color:#2c3e50">{d_str}</td>
          <td style="padding:6px 10px;font-size:11px;color:gray;text-align:center">{d_lbl}</td>
          <td style="padding:6px 10px;font-size:12px">{ev["icon"]} {ev["name"]}</td>
          <td style="padding:6px 10px">
            <span style="background:{col};color:white;font-size:11px;
                         padding:2px 8px;border-radius:10px">{ev["category"]}</span>
          </td>
          <td style="padding:6px 10px;font-size:11px;color:#7f8c8d">{ev["source"]}</td>
        </tr>'''

    return f'''
    <div style="background:#eaf4fb;border-radius:6px;padding:14px;margin-bottom:14px">
      <h3 style="margin:0 0 10px;font-size:14px;color:#2c3e50">🌍 향후 60일 거시경제 주요 일정</h3>
      <table style="border-collapse:collapse;width:100%">
        <thead>
          <tr style="font-size:12px;color:#7f8c8d;border-bottom:2px solid #bdc3c7">
            <th style="padding:5px 10px;text-align:left">날짜</th>
            <th style="padding:5px 10px">D-Day</th>
            <th style="padding:5px 10px;text-align:left">이벤트</th>
            <th style="padding:5px 10px">카테고리</th>
            <th style="padding:5px 10px;text-align:left">출처</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="font-size:11px;color:#95a5a6;margin:8px 0 0">
        ※ 2026년 일정은 추정치입니다. 실제 날짜와 ±1~2일 차이 가능. 투자 전 공식 사이트에서 확인 권장.
      </p>
    </div>'''


def _fetch_fed_rate_live():
    """FRED에서 Fed funds target rate 실시간 조회. (rate_str, date_str, success) 반환."""
    import urllib.request
    try:
        req = urllib.request.Request(
            'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU',
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            lines = r.read().decode('utf-8').strip().split('\n')
        for line in reversed(lines):
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    hi = float(parts[1])
                    lo = round(hi - 0.25, 2)
                    return f'{lo:.2f}% ~ {hi:.2f}%', parts[0], True
                except ValueError:
                    continue
    except Exception:
        pass
    return None, None, False


def _global_rates_block():
    """Fed금리·10Y국채·VIX·달러 실시간 표시 (세션 내 1시간 캐시)"""
    global _RATE_CACHE
    now = pd.Timestamp.now()
    if _RATE_CACHE.get('ts') and (now - _RATE_CACHE['ts']).seconds < 3600:
        return _RATE_CACHE['html']

    live = {}
    for label, sym in [('10Y 국채', '^TNX'), ('VIX', '^VIX'), ('달러(DXY)', 'DX-Y.NYB')]:
        try:
            h = yf.Ticker(sym).history(period='5d')['Close'].dropna()
            live[label] = round(float(h.iloc[-1]), 2) if not h.empty else None
        except Exception:
            live[label] = None

    # ── Fed 기준금리: FRED 실시간 조회 → 실패 시 하드코딩 + 신선도 경고 ──────
    fed_str, fed_date, fed_live = _fetch_fed_rate_live()
    if fed_live:
        fed_display = fed_str
        fed_note    = f'FRED 실시간 ({fed_date})'
        fed_col     = '#27ae60'
    else:
        fed_display = FED_RATE_RANGE
        try:
            days_old = (now - pd.Timestamp(FED_RATE_DATE)).days
            if days_old > 90:
                fed_note = f'⚠️ 하드코딩 ({FED_RATE_DATE} 기준, {days_old}일 경과 — 코드 업데이트 권장)'
                fed_col  = '#e74c3c'
            else:
                fed_note = f'하드코딩 ({FED_RATE_DATE} 기준)'
                fed_col  = '#f39c12'
        except Exception:
            fed_note = '하드코딩 기준 (FED_RATE_DATE 확인 필요)'
            fed_col  = '#f39c12'

    def _fmt(v, suffix=''):
        return f'{v}{suffix}' if v is not None else 'N/A'

    t10  = live.get('10Y 국채')
    vix  = live.get('VIX')
    dxy  = live.get('달러(DXY)')

    t10_col = '#e74c3c' if (t10 and t10 >= 4.5) else '#f39c12' if (t10 and t10 >= 3.5) else '#27ae60'
    t10_lbl = '고금리 ⚠️' if (t10 and t10 >= 4.5) else '보통' if (t10 and t10 >= 3.5) else '저금리'

    vix_col = '#c0392b' if (vix and vix >= 35) else '#e74c3c' if (vix and vix >= 25) else '#f39c12' if (vix and vix >= 15) else '#27ae60'
    vix_lbl = '극도공포' if (vix and vix >= 35) else '공포 ⚠️' if (vix and vix >= 25) else '주의' if (vix and vix >= 15) else '안정'

    rows = f'''
    <tr>
      <td style="padding:7px 14px;font-size:13px;color:#555"><b>🏦 미 연준 기준금리</b></td>
      <td style="padding:7px;font-size:14px;font-weight:bold;color:{fed_col}">{fed_display}</td>
      <td style="padding:7px 14px;font-size:12px;color:#7f8c8d">{fed_note}</td>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:7px 14px;font-size:13px;color:#555"><b>📈 미국 10Y 국채금리</b></td>
      <td style="padding:7px;font-size:14px;font-weight:bold;color:{t10_col}">{_fmt(t10, '%')}</td>
      <td style="padding:7px 14px;font-size:12px;color:#7f8c8d">{t10_lbl} — 4.5% 이상 시 성장주 밸류에이션 부담</td>
    </tr>
    <tr>
      <td style="padding:7px 14px;font-size:13px;color:#555"><b>😱 VIX 변동성 지수</b></td>
      <td style="padding:7px;font-size:14px;font-weight:bold;color:{vix_col}">{_fmt(vix)}</td>
      <td style="padding:7px 14px;font-size:12px;color:#7f8c8d">{vix_lbl} — 20↑ 주의 / 30↑ 공포 구간</td>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:7px 14px;font-size:13px;color:#555"><b>💵 달러 인덱스 (DXY)</b></td>
      <td style="padding:7px;font-size:14px;font-weight:bold;color:#2c3e50">{_fmt(dxy)}</td>
      <td style="padding:7px 14px;font-size:12px;color:#7f8c8d">달러 강세 시 신흥국·원자재·해외매출 기업 역풍</td>
    </tr>'''

    html = f'''
    <div style="background:#fff8e1;border-left:5px solid #f39c12;
                border-radius:6px;padding:14px;margin-bottom:14px">
      <h3 style="margin:0 0 10px;font-size:14px;color:#2c3e50">🌐 글로벌 금리 & 시장 지표 (실시간)</h3>
      <table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>
      <p style="font-size:11px;color:#95a5a6;margin:6px 0 0">
        ※ 연준 기준금리: FRED 실시간 조회 우선 (실패 시 코드 상수로 대체). 국채·VIX·DXY는 yfinance 실시간.
      </p>
    </div>'''

    _RATE_CACHE['html'] = html
    _RATE_CACHE['ts']   = now
    return html


def _insider_block(stock):
    """내부자 거래 최근 10건 표시"""
    try:
        it = stock.insider_transactions if stock else None
        if it is None or not isinstance(it, pd.DataFrame) or it.empty:
            return ''

        recent = it.head(10).copy()

        rows = ''
        for _, r in recent.iterrows():
            name  = str(r.get('Insider',     r.get('Name',     'N/A')))
            title = str(r.get('Title',       r.get('Position', '')))[:22]
            tx    = str(r.get('Transaction', r.get('Type',     '')))
            val   = r.get('Value',  r.get('Amount', None))
            date  = str(r.get('Start Date',  r.get('Date',     '')))[:10]

            is_buy  = any(w in tx for w in ('Purchase','Buy','Acquisition','Automatic Buy'))
            is_sell = any(w in tx for w in ('Sale','Sell','Disposition','Automatic Sell'))
            tx_col  = '#27ae60' if is_buy else '#e74c3c' if is_sell else '#7f8c8d'
            tx_icon = '▲ 매수' if is_buy else '▼ 매도' if is_sell else tx[:8]

            try:
                val_f   = float(val) if val is not None else None
                val_str = (f'${val_f/1e6:.1f}M' if val_f and abs(val_f) >= 1e6
                           else f'${val_f:,.0f}' if val_f else 'N/A')
            except Exception:
                val_str = 'N/A'

            rows += f'''
            <tr style="border-bottom:1px solid #ecf0f1">
              <td style="padding:5px 10px;font-size:12px;white-space:nowrap">{date}</td>
              <td style="padding:5px 10px;font-size:12px;font-weight:bold">{name}</td>
              <td style="padding:5px 10px;font-size:11px;color:#7f8c8d">{title}</td>
              <td style="padding:5px 10px;font-size:12px;font-weight:bold;
                         color:{tx_col};white-space:nowrap">{tx_icon}</td>
              <td style="padding:5px 10px;font-size:12px;text-align:right">{val_str}</td>
            </tr>'''

        if not rows:
            return ''

        return f'''
        <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
          <h3 style="margin:0 0 10px;font-size:14px;color:#2c3e50">👤 내부자 거래 (최근 10건)</h3>
          <table style="border-collapse:collapse;width:100%;
                        border:1px solid #ecf0f1;border-radius:4px;overflow:hidden">
            <thead>
              <tr style="background:#2c3e50;color:white;font-size:12px">
                <th style="padding:6px 10px;text-align:left">날짜</th>
                <th style="padding:6px 10px;text-align:left">임직원</th>
                <th style="padding:6px 10px;text-align:left">직책</th>
                <th style="padding:6px 10px">구분</th>
                <th style="padding:6px 10px">금액</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <p style="font-size:11px;color:#95a5a6;margin:6px 0 0">
            ※ 임원의 자사주 <span style="color:#27ae60;font-weight:bold">▲매수</span> =
            내부 신뢰 신호 &nbsp;|&nbsp;
            대량 <span style="color:#e74c3c;font-weight:bold">▼매도</span> = 주의 신호
          </p>
        </div>'''
    except Exception:
        return ''


def _eps_block(next_eps_date, eps_history, analyst, info):
    n = analyst['n_analysts']
    if n == 0:
        cov_col, cov_msg = '#e74c3c', '🔴 애널리스트 커버리지 없음 — 데이터 위험'
    elif n < 3:
        cov_col, cov_msg = '#f39c12', f'🟡 커버리지 부족 ({n}명) — 신뢰도 낮음'
    else:
        cov_col, cov_msg = '#27ae60', f'🟢 애널리스트 {n}명 — 신뢰도 양호'

    rec_map = {
        'strongBuy':  ('강력 매수', '#27ae60'),
        'buy':        ('매수',     '#2ecc71'),
        'hold':       ('중립',     '#f39c12'),
        'sell':       ('매도',     '#e74c3c'),
        'strongSell': ('강력 매도','#c0392b'),
    }
    rec_label, rec_color = rec_map.get(
        analyst['rec_key'], (analyst['rec_key'] or 'N/A', '#95a5a6'))

    def _fmt_price(v):
        return f'${v:.2f}' if v else 'N/A'

    t_mean_raw = analyst['target_mean']
    t_high     = _fmt_price(analyst['target_high'])
    t_low      = _fmt_price(analyst['target_low'])
    earnings_date_str = next_eps_date or analyst['earnings_date']

    # 목표가 업사이드 %
    current = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    if t_mean_raw and current:
        upside = (t_mean_raw - current) / current * 100
        up_sym = '▲' if upside >= 0 else '▼'
        up_col = '#27ae60' if upside >= 0 else '#e74c3c'
        t_mean_str = f'${t_mean_raw:.2f} <span style="color:{up_col};font-size:12px">({up_sym}{abs(upside):.1f}%)</span>'
    else:
        t_mean_str = _fmt_price(t_mean_raw)

    # Forward P/E vs Trailing P/E
    fwd_pe   = info.get('forwardPE')
    trail_pe = info.get('trailingPE')
    pe_str   = f'Forward {fwd_pe:.1f} / Trailing {trail_pe:.1f}' if fwd_pe and trail_pe else \
               f'Forward {fwd_pe:.1f}' if fwd_pe else _fmt_price(trail_pe)
    pe_col   = '#27ae60' if (fwd_pe and fwd_pe <= 25) else '#e74c3c' if (fwd_pe and fwd_pe > 35) else '#f39c12'

    # 공매도 비율
    short_pct = info.get('shortPercentOfFloat')
    if short_pct:
        short_str = f'{short_pct * 100:.1f}%'
        short_col = '#e74c3c' if short_pct > 0.10 else '#f39c12' if short_pct > 0.05 else '#27ae60'
        short_lbl = ' (위험 10%↑)' if short_pct > 0.10 else ' (주의 5%↑)' if short_pct > 0.05 else ''
    else:
        short_str, short_col, short_lbl = 'N/A', 'gray', ''

    # 52주 가격 위치
    h52 = info.get('fiftyTwoWeekHigh')
    l52 = info.get('fiftyTwoWeekLow')
    if h52 and l52 and current and h52 != l52:
        pos = (current - l52) / (h52 - l52) * 100
        pos_str = f'52주 {pos:.0f}% 위치  (저점 ${l52:.1f} ~ 고점 ${h52:.1f})'
        pos_col = '#27ae60' if pos > 60 else '#e74c3c' if pos < 30 else '#f39c12'
    else:
        pos_str, pos_col = 'N/A', 'gray'

    eps_rows = ''
    for q in eps_history:
        surp = q['surprise']
        if surp is not None:
            s_str = f"{'▲' if surp > 0 else '▼'} {abs(surp):.1f}%"
            s_col = '#27ae60' if surp > 0 else '#e74c3c'
        else:
            s_str, s_col = 'N/A', 'gray'
        eps_rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
          <td style="padding:6px 10px;font-size:12px">{q["date"]}</td>
          <td style="padding:6px 10px;text-align:right;font-size:12px">{q["estimate"]}</td>
          <td style="padding:6px 10px;text-align:right;font-size:12px;font-weight:bold">{q["actual"]}</td>
          <td style="padding:6px 10px;text-align:center;font-size:12px;font-weight:bold;color:{s_col}">{s_str}</td>
        </tr>'''
    if not eps_rows:
        eps_rows = '<tr><td colspan="4" style="padding:8px;color:#95a5a6;font-size:12px">EPS 이력 없음</td></tr>'

    return f'''
    <div style="background:#eaf4fb;border-radius:6px;padding:14px;margin-bottom:14px">
      <h3 style="margin:0 0 10px;font-size:14px;color:#2c3e50">📅 실적 캘린더 & 월가 컨센서스</h3>
      <table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:12px">
        <tr>
          <td style="padding:5px 14px;color:#555"><b>다음 실적 발표일</b></td>
          <td style="padding:5px;font-weight:bold;color:#2980b9">{earnings_date_str}</td>
          <td style="padding:5px 14px;color:#555"><b>투자 의견</b></td>
          <td style="padding:5px;font-weight:bold;color:{rec_color}">{rec_label}</td>
        </tr>
        <tr style="background:#f8f9fa">
          <td style="padding:5px 14px;color:#555"><b>평균 목표가</b></td>
          <td style="padding:5px;font-weight:bold">{t_mean_str}</td>
          <td style="padding:5px 14px;color:#555"><b>목표가 범위</b></td>
          <td style="padding:5px;font-size:12px">{t_low} ~ {t_high}</td>
        </tr>
        <tr>
          <td style="padding:5px 14px;color:#555"><b>PER (Forward/Trailing)</b></td>
          <td style="padding:5px;font-weight:bold;color:{pe_col}">{pe_str}</td>
          <td style="padding:5px 14px;color:#555"><b>공매도 비율</b></td>
          <td style="padding:5px;font-weight:bold;color:{short_col}">{short_str}{short_lbl}</td>
        </tr>
        <tr style="background:#f8f9fa">
          <td style="padding:5px 14px;color:#555"><b>52주 가격 위치</b></td>
          <td colspan="3" style="padding:5px;font-size:12px;color:{pos_col};font-weight:bold">{pos_str}</td>
        </tr>
      </table>
      <p style="margin:0 0 8px;font-size:12px;font-weight:bold;color:{cov_col}">{cov_msg}</p>
      <details open>
        <summary style="cursor:pointer;font-size:12px;color:#2980b9;font-weight:bold;margin-bottom:8px">
          분기별 EPS 실적 이력 (최근 4분기)
        </summary>
        <table style="border-collapse:collapse;width:100%;margin-top:6px;
                      border:1px solid #ecf0f1;border-radius:4px;overflow:hidden">
          <thead>
            <tr style="background:#2c3e50;color:white;font-size:12px">
              <th style="padding:6px 10px;text-align:left">분기</th>
              <th style="padding:6px 10px">예상 EPS</th>
              <th style="padding:6px 10px">실제 EPS</th>
              <th style="padding:6px 10px">서프라이즈</th>
            </tr>
          </thead>
          <tbody>{eps_rows}</tbody>
        </table>
      </details>
    </div>'''


def _sector_risk_block(sector, risks):
    rows = ''
    for i, (factor, desc) in enumerate(risks):
        bg = '#fef9f9' if i % 2 == 0 else '#fff'
        rows += f'''
        <tr style="background:{bg};border-bottom:1px solid #ecf0f1">
          <td style="padding:7px 12px;font-size:12px;font-weight:bold;
                     color:#c0392b;white-space:nowrap">⚠️ {factor}</td>
          <td style="padding:7px 12px;font-size:12px;color:#555">{desc}</td>
        </tr>'''

    return f'''
    <div style="background:#fef5f5;border-left:5px solid #e74c3c;
                border-radius:6px;padding:14px;margin-bottom:14px">
      <h3 style="margin:0 0 10px;font-size:14px;color:#c0392b">
        ⚠️ 섹터 리스크 요인 — {sector or "공통"}
      </h3>
      <table style="border-collapse:collapse;width:100%">
        <tbody>{rows}</tbody>
      </table>
    </div>'''


def _scenarios(info):
    roe        = (info.get('returnOnEquity')               or 0) * 100
    per        =  info.get('trailingPE')                   or 0
    psr        =  info.get('priceToSalesTrailing12Months') or 0
    rev_growth = (info.get('revenueGrowth')                or 0) * 100
    eps_growth = (info.get('earningsGrowth')               or 0) * 100
    target     =  info.get('targetMeanPrice')              or 0
    current    =  info.get('currentPrice') or info.get('regularMarketPrice') or 0
    upside     = ((target - current) / current * 100) if (current and target) else 0

    bulls = []
    if roe >= 15:        bulls.append(f'ROE {roe:.1f}% 유지 → 자본 효율성이 밸류에이션 프리미엄 정당화')
    if rev_growth >= 15: bulls.append(f'매출 {rev_growth:.1f}% 고성장 → 성장주 재평가 랠리 기대')
    if eps_growth > 10:  bulls.append(f'EPS {eps_growth:.1f}% 성장 → 수익성 개선이 주가 모멘텀 연결')
    if upside > 15:      bulls.append(f'월가 평균 목표가 ${target:.2f} → +{upside:.1f}% 상승 여력')
    if not bulls:        bulls.append('재무 기반 강세 트리거 불명확 — 정성적 분석 병행 권장')

    bears = []
    if per > 30:         bears.append(f'PER {per:.1f} 고평가 → 실적 미스 시 밸류에이션 압축 낙폭 위험')
    if psr > 5:          bears.append(f'PSR {psr:.1f} 과열 → 성장 둔화 시 급격한 재평가 리스크')
    if 0 < roe < 10:     bears.append(f'ROE {roe:.1f}% 저조 → 자본 비효율로 경쟁사 대비 구조적 열위')
    if rev_growth < 0:   bears.append(f'매출 역성장 {rev_growth:.1f}% → 핵심 사업 수요 감소 신호')
    if upside < -5:      bears.append(f'목표가 ${target:.2f} → 현재가 대비 하락 여지, 월가 센티멘트 부정적')
    if not bears:        bears.append('재무 기반 약세 트리거 불명확 — 거시 환경 모니터링 권장')

    return bulls[:2], bears[:2]


def _news_block(news_items):
    if not news_items:
        return '''
        <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
          <h3 style="margin:0 0 6px;font-size:14px;color:#2c3e50">📰 최근 뉴스</h3>
          <p style="margin:0;font-size:12px;color:#95a5a6">뉴스 데이터 없음</p>
        </div>'''

    items_html = ''
    for i, item in enumerate(news_items, 1):
        short = item['title'][:85] + '...' if len(item['title']) > 85 else item['title']
        link_btn = (
            f'<a href="{item["link"]}" target="_blank" '
            f'style="display:inline-block;background:#2980b9;color:white;'
            f'font-size:12px;padding:4px 14px;border-radius:4px;'
            f'text-decoration:none;font-weight:bold">원문 보기 →</a>'
            if item['link'] else
            '<span style="font-size:12px;color:#95a5a6">링크 없음</span>'
        )
        items_html += f'''
        <details style="margin-bottom:5px;border:1px solid #dde4e9;
                        border-radius:6px;overflow:hidden">
          <summary style="cursor:pointer;padding:10px 14px;background:#f8f9fa;
                          list-style:none;outline:none;display:block">
            <span style="font-size:12px;font-weight:bold;color:#3498db">#{i}</span>
            &nbsp;
            <span style="font-size:13px;color:#2c3e50">{short}</span>
            <span style="float:right;font-size:11px;color:#95a5a6;white-space:nowrap;margin-left:8px">
              {item["source"]} · {item["date"]}
            </span>
          </summary>
          <div style="padding:12px 16px;background:#fff;border-top:1px solid #ecf0f1">
            <p style="margin:0 0 8px;font-size:13px;color:#2c3e50;line-height:1.7">
              {item["title"]}
            </p>
            <div style="font-size:12px;color:#7f8c8d;margin-bottom:10px">
              출처: <b>{item["source"]}</b> &nbsp;|&nbsp; 날짜: <b>{item["date"]}</b>
            </div>
            {link_btn}
          </div>
        </details>'''

    return f'''
    <div style="margin-bottom:14px">
      <h3 style="margin:0 0 8px;color:#2c3e50;font-size:14px">
        📰 최근 뉴스 (최대 6개) — 클릭하면 펼쳐집니다
      </h3>
      {items_html}
      <p style="font-size:11px;color:#95a5a6;margin-top:4px">
        * 제목 클릭으로 펼치기 / 다시 클릭으로 접기
      </p>
    </div>'''


# ══════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════

def analyze_news(ticker, info, raw_news,
                 raw_dividends=None, raw_eps=None, stock=None):
    # ── Bug fix 2: summary가 None일 때 len() 오류 방지 ──────────
    summary = info.get('longBusinessSummary') or '기업 정보 없음'
    summary = summary[:250] + '...' if len(summary) > 250 else summary

    name     = info.get('longName') or info.get('shortName') or ticker
    sector   = info.get('sector') or ''
    industry = info.get('industry') or 'N/A'

    analyst          = _analyst_info(stock, info)
    next_eps_dt, eps_hist = _parse_eps(raw_eps)
    news_items       = _parse_news(raw_news)
    upcoming         = _upcoming_macro(days_ahead=60)
    risks            = SECTOR_RISKS.get(sector, DEFAULT_RISKS)[:4]
    bulls, bears     = _scenarios(info)

    bull_li = ''.join(f'<li style="margin:5px 0;font-size:13px">✅ {b}</li>' for b in bulls)
    bear_li = ''.join(f'<li style="margin:5px 0;font-size:13px">⚠️ {b}</li>' for b in bears)

    html = f'''
    <div style="font-family:Arial,sans-serif;max-width:880px;margin-bottom:20px">
      <h2 style="color:#2c3e50;border-bottom:3px solid #3498db;
                 padding-bottom:8px;margin-bottom:14px">
        📰 STEP 3 — [{ticker}] 뉴스 & 시나리오 분석
      </h2>

      <div style="background:#f4f6f7;border-radius:6px;padding:14px;margin-bottom:14px">
        <h3 style="margin:0 0 6px;color:#2c3e50;font-size:14px">🏢 기업 소개</h3>
        <p style="margin:0;font-size:13px"><b>{name}</b> &nbsp;|&nbsp; {sector} › {industry}</p>
        <p style="margin:8px 0 0;font-size:12px;color:#555;line-height:1.7">{summary}</p>
      </div>

      {_eps_block(next_eps_dt, eps_hist, analyst, info)}
      {_dividend_block(info, raw_dividends)}
      {_insider_block(stock)}
      {_global_rates_block()}
      {_macro_block(upcoming)}
      {_sector_risk_block(sector, risks)}
      {_news_block(news_items)}

      <div style="display:flex;gap:12px;margin-bottom:14px">
        <div style="flex:1;background:#eafaf1;border-left:5px solid #27ae60;
                    border-radius:6px;padding:14px">
          <h3 style="margin:0 0 8px;color:#27ae60;font-size:14px">🟢 강세 (Bull) 시나리오</h3>
          <ul style="margin:0;padding-left:16px;line-height:1.8">{bull_li}</ul>
        </div>
        <div style="flex:1;background:#fef5f5;border-left:5px solid #e74c3c;
                    border-radius:6px;padding:14px">
          <h3 style="margin:0 0 8px;color:#e74c3c;font-size:14px">🔴 약세 (Bear) 시나리오</h3>
          <ul style="margin:0;padding-left:16px;line-height:1.8">{bear_li}</ul>
        </div>
      </div>

      <p style="font-size:11px;color:#95a5a6;border-top:1px solid #ecf0f1;padding-top:8px">
        ⚠️ 본 분석은 정보 제공 목적이며 투자 권유가 아닙니다.
        시나리오는 재무 수치 기반 규칙으로 자동 생성되며 실제 시장 결과를 보장하지 않습니다.<br>
        분석 기준일: {pd.Timestamp.now().strftime('%Y-%m-%d')} &nbsp;|&nbsp; 데이터 출처: yfinance
      </p>
    </div>
    '''
    display(HTML(html))
