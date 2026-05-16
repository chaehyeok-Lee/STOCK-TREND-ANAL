import pandas as pd
from IPython.display import display, HTML
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── 섹터/산업 → ETF 매핑 ─────────────────────────────────────────────────────

_INDUSTRY_ETF = {
    'Biotechnology':                          ('IBB',  'Nasdaq Biotech',     ['AMGN','GILD','VRTX','REGN','BIIB','MRNA','ILMN','INCY','ALNY','BMRN']),
    'Drug Manufacturers—General':             ('XPH',  'S&P Pharma',         ['LLY','ABBV','MRK','PFE','BMY','JNJ','NVO','AZN','GSK','SNY']),
    'Drug Manufacturers—Specialty & Generic': ('XPH',  'S&P Pharma',         ['LLY','ABBV','MRK','PFE','BMY','JNJ','NVO','AZN','GSK','SNY']),
    'Semiconductors':                         ('SOXX', 'iShares Semi',        ['NVDA','AVGO','QCOM','TXN','AMD','INTC','AMAT','KLAC','LRCX','MCHP']),
    'Semiconductor Equipment & Materials':    ('SOXX', 'iShares Semi',        ['NVDA','AVGO','QCOM','TXN','AMD','INTC','AMAT','KLAC','LRCX','MCHP']),
    'Software—Application':                   ('IGV',  'iShares Software',    ['MSFT','ORCL','CRM','SAP','ADBE','NOW','WDAY','INTU','TEAM','HUBS']),
    'Software—Infrastructure':                ('IGV',  'iShares Software',    ['MSFT','ORCL','CRM','SAP','ADBE','NOW','WDAY','INTU','TEAM','HUBS']),
    'Banks—Diversified':                      ('KBE',  'S&P Bank',            ['JPM','BAC','WFC','C','USB','TFC','KEY','CFG','FITB','HBAN']),
    'Banks—Regional':                         ('KRE',  'S&P Regional Bank',   ['USB','TFC','KEY','CFG','FITB','HBAN','MTB','RF','ZION','CMA']),
    'Oil & Gas E&P':                          ('XOP',  'Oil & Gas E&P',       ['XOM','CVX','COP','EOG','DVN','APA','MRO','CTRA','OVV','SM']),
    'Oil & Gas Integrated':                   ('XLE',  'Energy Select',       ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL']),
    'Medical Devices':                        ('IHI',  'iShares Med. Dev.',   ['MDT','ABT','SYK','BSX','EW','ZBH','ISRG','RMD','DXCM','PODD']),
    'Diagnostics & Research':                 ('IHI',  'iShares Med. Dev.',   ['MDT','ABT','SYK','BSX','EW','ZBH','ISRG','RMD','DXCM','PODD']),
    'Auto Manufacturers':                     ('CARZ', 'Global Auto',         ['TSLA','TM','GM','F','STLA','HMC','VWAGY','BMWYY','RACE','NIO']),
    'Internet Content & Information':         ('XWEB', 'SPDR Internet',       ['META','GOOGL','NFLX','SNAP','PINS','UBER','LYFT','YELP','IAC','MTCH']),
}

_SECTOR_ETF = {
    'Technology':             ('XLK',  'Tech Select',        ['AAPL','MSFT','NVDA','AVGO','ORCL','CSCO','CRM','ACN','ADBE','AMD']),
    'Healthcare':             ('XLV',  'Healthcare Select',  ['UNH','LLY','JNJ','ABBV','MRK','TMO','ABT','PFE','DHR','CVS']),
    'Financial Services':     ('XLF',  'Financial Select',   ['BRK-B','JPM','V','MA','BAC','WFC','GS','MS','C','AXP']),
    'Energy':                 ('XLE',  'Energy Select',      ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL']),
    'Consumer Cyclical':      ('XLY',  'Consumer Discret.',  ['AMZN','TSLA','HD','MCD','NKE','SBUX','TJX','LOW','BKNG','CMG']),
    'Consumer Defensive':     ('XLP',  'Consumer Staples',   ['PG','COST','KO','PEP','WMT','PM','MO','CL','MDLZ','GIS']),
    'Industrials':            ('XLI',  'Industrials Select', ['RTX','HON','UPS','UNP','LMT','CAT','DE','GE','ETN','MMM']),
    'Real Estate':            ('XLRE', 'Real Estate Select', ['AMT','PLD','CCI','EQIX','PSA','DLR','O','SPG','AVB','EQR']),
    'Communication Services': ('XLC',  'Comm. Services',     ['META','GOOGL','NFLX','CMCSA','T','VZ','TMUS','DIS','EA','WBD']),
    'Utilities':              ('XLU',  'Utilities Select',   ['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ED','WEC']),
    'Basic Materials':        ('XLB',  'Materials Select',   ['LIN','APD','SHW','FCX','NEM','NUE','ALB','DD','ECL','VMC']),
}

_HIGHER_BETTER = {'ROE', 'RevGrowth'}


def _get_sector_etf(sector, industry):
    if industry and industry in _INDUSTRY_ETF:
        return _INDUSTRY_ETF[industry]
    if sector and sector in _SECTOR_ETF:
        return _SECTOR_ETF[sector]
    return None


def _fetch_one(t):
    try:
        return t, yf.Ticker(t).info
    except Exception:
        return t, {}


def _peer_avgs(peers, exclude):
    targets = [t for t in peers if t.upper() != exclude.upper()]
    infos = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in targets}
        for f in as_completed(futures):
            t, info = f.result()
            infos[t] = info

    def avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 2) if v else None

    roes, pers, psrs, pbrs, revgs, des = [], [], [], [], [], []
    for info in infos.values():
        roes.append(_safe(info, 'returnOnEquity', mult=100))
        pers.append(_safe(info, 'trailingPE'))
        psrs.append(_safe(info, 'priceToSalesTrailing12Months'))
        pbrs.append(_safe(info, 'priceToBook'))
        revgs.append(_safe(info, 'revenueGrowth', mult=100))
        raw_de = _safe(info, 'debtToEquity')
        des.append(round(raw_de / 100, 2) if raw_de is not None else None)

    return {
        'ROE': avg(roes), 'PER': avg(pers), 'PSR': avg(psrs),
        'PBR': avg(pbrs), 'RevGrowth': avg(revgs), 'DE': avg(des),
    }


def _compare(val, avg_val, metric):
    if val is None or avg_val is None:
        return '—', '#bdc3c7'
    if avg_val == 0:
        return '≈ 유사', '#7f8c8d'
    diff = (val - avg_val) / abs(avg_val) * 100
    if abs(diff) < 10:
        return '≈ 유사', '#7f8c8d'
    above = diff > 0
    if metric in _HIGHER_BETTER:
        return ('▲ 상위', '#27ae60') if above else ('▼ 하위', '#e74c3c')
    else:
        return ('▲ 우위', '#27ae60') if not above else ('▼ 열위', '#e74c3c')


# ── 기준 평가 ────────────────────────────────────────────────────────────────

def _safe(info, key, mult=1, dec=2):
    v = info.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return None
    return round(float(v) * mult, dec)


def _grade(val, metric):
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


# ── 메인 함수 ────────────────────────────────────────────────────────────────

def analyze_financials(ticker, info):
    roe        = _safe(info, 'returnOnEquity', mult=100)
    per        = _safe(info, 'trailingPE')
    psr        = _safe(info, 'priceToSalesTrailing12Months')
    pbr        = _safe(info, 'priceToBook')
    rev_growth = _safe(info, 'revenueGrowth', mult=100)
    raw_de     = _safe(info, 'debtToEquity')
    de         = round(raw_de / 100, 2) if raw_de is not None else None

    metrics = [
        ('ROE (%)',           'ROE',       roe,        '자기자본이익률 — 높을수록 수익성 우수'),
        ('PER',               'PER',       per,        '주가수익비율 — 낮을수록 저평가'),
        ('PSR',               'PSR',       psr,        '주가매출비율 — 낮을수록 저평가'),
        ('PBR',               'PBR',       pbr,        '주가순자산비율 — 낮을수록 저평가'),
        ('매출 성장률 YoY (%)', 'RevGrowth', rev_growth, '전년 대비 매출 성장률'),
        ('부채비율 D/E',        'DE',        de,         '부채÷자본 — 낮을수록 재무 안전'),
    ]

    # ── 섹터 ETF 피어 평균 로딩 ───────────────────────────────
    sector   = info.get('sector', '')
    industry = info.get('industry', '')
    etf_info = _get_sector_etf(sector, industry)
    etf_ticker = etf_name = ''
    etf_avgs = None
    if etf_info:
        etf_ticker, etf_name, peers = etf_info
        print(f'  ⏳ [{etf_ticker}] 섹터 피어 평균 로딩 중 ({len(peers)-1}개 종목)...')
        etf_avgs = _peer_avgs(peers, ticker)
        print(f'  ✅ [{etf_ticker}] 피어 평균 로딩 완료')

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

        if etf_avgs is not None:
            avg_val = etf_avgs.get(key)
            avg_str = 'N/A' if avg_val is None else str(avg_val)
            cmp_label, cmp_color = _compare(val, avg_val, key)
            etf_cell = f'''
            <td style="padding:8px 10px;text-align:center;border-left:1px solid #ecf0f1;white-space:nowrap">
                <div style="font-size:13px;font-weight:bold;color:#2c3e50">{avg_str}</div>
                <div style="font-size:11px;font-weight:bold;color:{cmp_color};margin-top:2px">{cmp_label}</div>
            </td>'''
        else:
            etf_cell = ''

        rows += f'''
        <tr style="border-bottom:1px solid #ecf0f1">
            <td style="padding:9px 12px;font-weight:bold;color:#2c3e50">{label}</td>
            <td style="padding:9px 12px;text-align:center;font-size:14px;font-weight:bold">{val_str}</td>
            <td style="padding:6px 10px;text-align:center">
                <span style="background:{color};color:white;border-radius:5px;padding:3px 10px;font-size:12px;font-weight:bold">{status}</span>
            </td>{etf_cell}
            <td style="padding:9px 12px;font-size:11px;color:#7f8c8d">{desc}</td>
        </tr>'''

    name   = info.get('longName') or info.get('shortName') or ticker
    price  = info.get('currentPrice') or info.get('regularMarketPrice') or 'N/A'
    mcap   = info.get('marketCap')
    mcap_s = f"${mcap/1e9:.1f}B" if mcap else 'N/A'

    if etf_avgs is not None:
        etf_header = f'<th style="padding:10px 12px;width:13%">[{etf_ticker}] 섹터평균</th>'
        peer_note  = f'섹터 피어: {etf_ticker} ({etf_name})'
        max_w = '1100px'
    else:
        etf_header = ''
        peer_note  = '섹터 피어: 미매핑'
        max_w = '860px'

    html = f'''
    <div style="font-family:Arial,sans-serif;max-width:{max_w};margin-bottom:30px">
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
            <th style="padding:10px 12px;text-align:left;width:17%">지표</th>
            <th style="padding:10px 12px;width:9%">값</th>
            <th style="padding:10px 12px;width:27%">평가</th>
            {etf_header}
            <th style="padding:10px 12px;text-align:left">설명</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="font-size:11px;color:#95a5a6;margin-top:6px">
        🟢 우수/저평가 &nbsp;|&nbsp; 🟡 보통 &nbsp;|&nbsp; 🔴 위험/고평가 &nbsp;|&nbsp;
        ▲ 우위 / ▼ 열위: 섹터 평균 대비 유리/불리한 방향 &nbsp;|&nbsp;
        {peer_note} &nbsp;|&nbsp; 데이터 출처: yfinance
      </p>
    </div>
    '''
    display(HTML(html))
