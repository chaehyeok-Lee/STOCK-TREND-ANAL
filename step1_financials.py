import pandas as pd
from IPython.display import display, HTML


def _safe(info, key, mult=1, dec=2):
    v = info.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return None
    return round(float(v) * mult, dec)


def _grade(val, metric):
    """(color, label, is_missing) 반환"""
    if val is None:
        return '#95a5a6', 'N/A', True

    if metric == 'ROE':
        if val >= 15:  return '#27ae60', f'우수 ({val}% ≥ 15%)', False
        if val >= 10:  return '#f39c12', f'보통 ({val}% / 10~15%)', False
        if val >= 0:   return '#e74c3c', f'위험 ({val}% < 10%)', True
        return             '#c0392b', f'위험 (음수 {val}%)', True

    if metric == 'PER':
        if val <= 0:   return '#c0392b', f'위험 (음수/비정상)', True
        if val <= 20:  return '#27ae60', f'저평가 ({val} ≤ 20)', False
        if val <= 30:  return '#f39c12', f'보통 ({val} / 20~30)', False
        return             '#e74c3c', f'고평가 ({val} > 30)', True

    if metric == 'PSR':
        if val <= 3:   return '#27ae60', f'저평가 ({val} ≤ 3)', False
        if val <= 5:   return '#f39c12', f'보통 ({val} / 3~5)', False
        return             '#e74c3c', f'고평가 ({val} > 5)', True

    if metric == 'PBR':
        if val <= 1.5: return '#27ae60', f'저평가 ({val} ≤ 1.5)', False
        if val <= 3:   return '#f39c12', f'보통 ({val} / 1.5~3)', False
        return             '#e74c3c', f'고평가 ({val} > 3)', True

    if metric == 'RevGrowth':
        if val >= 20:  return '#27ae60', f'고성장 ({val}% ≥ 20%)', False
        if val >= 5:   return '#f39c12', f'보통 ({val}% / 5~20%)', False
        if val >= 0:   return '#e74c3c', f'저성장 ({val}% < 5%)', True
        return             '#c0392b', f'역성장 ({val}%)', True

    if metric == 'DE':
        if val <= 1.0: return '#27ae60', f'안전 ({val} ≤ 1.0)', False
        if val <= 2.0: return '#f39c12', f'보통 ({val} / 1~2)', False
        return             '#e74c3c', f'위험 ({val} > 2.0)', True

    return '#95a5a6', str(val), False


def analyze_financials(ticker, info):
    roe        = _safe(info, 'returnOnEquity', mult=100)
    per        = _safe(info, 'trailingPE')
    psr        = _safe(info, 'priceToSalesTrailing12Months')
    pbr        = _safe(info, 'priceToBook')
    rev_growth = _safe(info, 'revenueGrowth', mult=100)
    raw_de     = _safe(info, 'debtToEquity')
    de         = round(raw_de / 100, 2) if raw_de is not None else None

    metrics = [
        ('ROE (%)',          'ROE',       roe,        '자기자본이익률 — 높을수록 수익성 우수'),
        ('PER',              'PER',       per,        '주가수익비율 — 낮을수록 저평가'),
        ('PSR',              'PSR',       psr,        '주가매출비율 — 낮을수록 저평가'),
        ('PBR',              'PBR',       pbr,        '주가순자산비율 — 낮을수록 저평가'),
        ('매출 성장률 YoY (%)', 'RevGrowth', rev_growth, '전년 대비 매출 성장률'),
        ('부채비율 D/E',       'DE',        de,         '부채÷자본 — 낮을수록 재무 안전'),
    ]

    missing = sum(1 for _, _, v, _ in metrics if v is None)
    avail   = 6 - missing

    if avail >= 5:
        rc, rl = '#27ae60', f'🟢 데이터 안전 — {avail}/6 지표 정상, 신뢰도 높음'
    elif avail >= 3:
        rc, rl = '#f39c12', f'🟡 데이터 주의 — {avail}/6 지표 정상, {missing}개 N/A'
    else:
        rc, rl = '#e74c3c', f'🔴 데이터 위험 — {avail}/6 지표만 확인됨, 투자 판단 주의 필요'

    rows = ''
    for label, key, val, desc in metrics:
        color, status, _ = _grade(val, key)
        val_str = 'N/A' if val is None else str(val)
        rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
            <td style="padding:9px 12px;font-weight:bold;color:#2c3e50">{label}</td>
            <td style="padding:9px 12px;text-align:center;font-size:14px;font-weight:bold">{val_str}</td>
            <td style="padding:6px 10px;text-align:center">
                <span style="background:{color};color:white;border-radius:5px;padding:3px 10px;font-size:12px;font-weight:bold">{status}</span>
            </td>
            <td style="padding:9px 12px;font-size:11px;color:#7f8c8d">{desc}</td>
        </tr>'''

    name    = info.get('longName') or info.get('shortName') or ticker
    sector  = info.get('sector', 'N/A')
    industry= info.get('industry', 'N/A')
    price   = info.get('currentPrice') or info.get('regularMarketPrice') or 'N/A'
    mcap    = info.get('marketCap')
    mcap_s  = f"${mcap/1e9:.1f}B" if mcap else 'N/A'

    html = f'''
    <div style="font-family:Arial,sans-serif;max-width:860px;margin-bottom:30px">
      <h2 style="color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:8px;margin-bottom:14px">
        📊 STEP 1 — [{ticker}] 재무 지표 분석
      </h2>
      <p style="margin:0 0 10px;font-size:13px;color:#555">
        <b>{name}</b> &nbsp;|&nbsp; 섹터: <b>{sector}</b> &nbsp;|&nbsp; 산업: {industry}<br>
        현재가: <b>${price}</b> &nbsp;|&nbsp; 시가총액: <b>{mcap_s}</b>
      </p>
      <div style="background:{rc}18;border-left:5px solid {rc};padding:10px 14px;border-radius:5px;margin-bottom:14px">
        <span style="color:{rc};font-weight:bold;font-size:13px">{rl}</span>
      </div>
      <table style="border-collapse:collapse;width:100%;background:#fff;border:1px solid #ecf0f1;border-radius:6px;overflow:hidden">
        <thead>
          <tr style="background:#2c3e50;color:white;font-size:13px">
            <th style="padding:10px 12px;text-align:left;width:18%">지표</th>
            <th style="padding:10px 12px;width:10%">값</th>
            <th style="padding:10px 12px;width:30%">평가</th>
            <th style="padding:10px 12px;text-align:left">설명</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="font-size:11px;color:#95a5a6;margin-top:6px">
        🟢 우수/저평가 &nbsp;|&nbsp; 🟡 보통 &nbsp;|&nbsp; 🔴 위험/고평가 &nbsp;|&nbsp;
        기준: 절대값 기준 적용 (섹터 평균 미반영) &nbsp;|&nbsp; 데이터 출처: yfinance
      </p>
    </div>
    '''
    display(HTML(html))
