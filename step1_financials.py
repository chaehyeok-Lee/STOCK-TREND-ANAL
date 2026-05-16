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

_HIGHER_BETTER = {'ROE', 'ROIC', 'RevGrowth', 'GrossM', 'OpM', 'NetM', 'FCFMargin', 'FCFYield', 'EpsGrowth', 'CurrRatio'}

# ── 섹터/산업별 밸류에이션 임계값 ─────────────────────────────────────────────
# PER/PSR/PBR: (good, ok) — ≤good=우수, ≤ok=보통, else=고평가
# ROE:         (good, ok) — ≥good=우수, ≥ok=보통, else=저조
# None → 해당 지표 비교 부적합 (예: 바이오 PER)
_SECTOR_THRESHOLDS: dict = {
    '__default__':            {'PER': (20,  30), 'PSR': (3,  5),  'PBR': (1.5, 3.0), 'ROE': (15, 10)},
    'Technology':             {'PER': (30,  50), 'PSR': (5, 10),  'PBR': (4,  10)},
    'Communication Services': {'PER': (25,  40), 'PSR': (4,  8),  'PBR': (3,   7)},
    'Healthcare':             {'PER': (25,  40), 'PSR': (3,  6),  'PBR': (3,   6)},
    'Financial Services':     {'PER': (15,  20), 'PSR': (2,  4),  'PBR': (1.2, 2.0), 'ROE': (12, 8)},
    'Energy':                 {'PER': (15,  20), 'PSR': (1,  2),  'PBR': (1.5, 2.5)},
    'Consumer Defensive':     {'PER': (22,  30), 'PSR': (1,  2),  'PBR': (3,   5)},
    'Utilities':              {'PER': (18,  25), 'PSR': (2,  3),  'PBR': (1.5, 2.5), 'ROE': (10, 6)},
    'Basic Materials':        {'PER': (15,  22), 'PSR': (1,  2),  'PBR': (1.5, 2.5)},
    'Industrials':            {'PER': (20,  30), 'PSR': (2,  4),  'PBR': (2,   4)},
    'Consumer Cyclical':      {'PER': (22,  35), 'PSR': (2,  4),  'PBR': (2,   5)},
    'Real Estate':            {'PER': (25,  40), 'PSR': (4,  8),  'PBR': (1.5, 3.0), 'ROE': (8, 4)},
}
_INDUSTRY_THRESHOLDS: dict = {
    'Biotechnology':                  {'PER': None,     'PSR': (6, 12), 'ROE': (0, -20)},
    'Drug Manufacturers—General':     {'PER': (22, 35)},
    'Software—Application':           {'PER': (35, 60), 'PSR': (6, 12), 'PBR': (5, 12)},
    'Software—Infrastructure':        {'PER': (35, 60), 'PSR': (6, 12), 'PBR': (5, 12)},
    'Internet Content & Information': {'PER': (30, 55), 'PSR': (5, 10), 'PBR': (4, 10)},
    'Semiconductors':                 {'PER': (25, 40), 'PSR': (4,  8)},
}


def _get_thresh(metric, sector='', industry=''):
    """섹터/산업별 임계값 반환. None = 해당 지표 비교 부적합."""
    ind = _INDUSTRY_THRESHOLDS.get(industry or '', {})
    if metric in ind:
        return ind[metric]
    sec = _SECTOR_THRESHOLDS.get(sector or '', {})
    if metric in sec:
        return sec[metric]
    return _SECTOR_THRESHOLDS['__default__'].get(metric)


def _get_sector_etf(sector, industry):
    if industry and industry in _INDUSTRY_ETF:
        return _INDUSTRY_ETF[industry]
    if sector and sector in _SECTOR_ETF:
        return _SECTOR_ETF[sector]
    return None


def _fetch_one(t):
    try:
        info = yf.Ticker(t).info
        return t, info if info else None
    except Exception:
        return t, None


def _peer_avgs(peers, exclude):
    targets = [t for t in peers if t.upper() != exclude.upper()]
    infos = {}
    failed = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in targets}
        for f in as_completed(futures):
            t, info = f.result()
            if info is not None:
                infos[t] = info
            else:
                failed.append(t)
    if failed:
        print(f'  ⚠️  피어 데이터 누락 ({len(failed)}/{len(targets)}개): {", ".join(failed)}')

    def avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 2) if v else None

    roes, pers, psrs, pbrs, revgs, des = [], [], [], [], [], []
    gross_ms, op_ms, net_ms, ev_ebitdas = [], [], [], []
    betas, curr_ratios, eps_growths = [], [], []
    fcf_yields, fcf_margins = [], []
    pegs, roics, net_debt_ebitdas = [], [], []

    for info in infos.values():
        roes.append(_safe(info, 'returnOnEquity', mult=100))
        pers.append(_safe(info, 'trailingPE'))
        psrs.append(_safe(info, 'priceToSalesTrailing12Months'))
        pbrs.append(_safe(info, 'priceToBook'))
        revgs.append(_safe(info, 'revenueGrowth', mult=100))
        raw_de = _safe(info, 'debtToEquity')
        des.append(round(raw_de / 100, 2) if raw_de is not None else None)
        gross_ms.append(_safe(info, 'grossMargins',     mult=100))
        op_ms.append(_safe(info, 'operatingMargins',    mult=100))
        net_ms.append(_safe(info, 'profitMargins',      mult=100))
        ev_ebitdas.append(_safe(info, 'enterpriseToEbitda'))
        betas.append(_safe(info, 'beta'))
        curr_ratios.append(_safe(info, 'currentRatio'))
        eps_growths.append(_safe(info, 'earningsGrowth', mult=100))
        fcf  = info.get('freeCashflow')
        mcap = info.get('marketCap')
        rev  = info.get('totalRevenue')
        fcf_yields.append(round(fcf / mcap * 100, 2) if (fcf and mcap and mcap > 0) else None)
        fcf_margins.append(round(fcf / rev  * 100, 2) if (fcf and rev  and rev  > 0) else None)
        # PEG
        _per = _safe(info, 'trailingPE')
        _eg  = _safe(info, 'earningsGrowth', mult=100)
        pegs.append(round(_per / _eg, 2) if (_per and _eg and _eg > 0 and _per > 0) else None)
        # ROIC
        _ebit = info.get('ebit')
        _tax  = info.get('effectiveTaxRate') or 0.21
        _eq   = info.get('totalStockholderEquity') or 0
        _debt = info.get('totalDebt') or 0
        _cash = info.get('totalCash') or 0
        _ic   = _eq + _debt - _cash
        roics.append(round(_ebit * (1 - _tax) / _ic * 100, 2) if (_ebit is not None and _ic > 0) else None)
        # Net Debt / EBITDA
        _ebitda = info.get('ebitda')
        net_debt_ebitdas.append(round((_debt - _cash) / _ebitda, 2) if (_ebitda and _ebitda > 0) else None)

    return {
        'ROE': avg(roes), 'PER': avg(pers), 'PSR': avg(psrs),
        'PBR': avg(pbrs), 'RevGrowth': avg(revgs), 'DE': avg(des),
        'GrossM': avg(gross_ms), 'OpM': avg(op_ms), 'NetM': avg(net_ms),
        'EV_EBITDA': avg(ev_ebitdas), 'Beta': avg(betas), 'CurrRatio': avg(curr_ratios),
        'FCFYield': avg(fcf_yields), 'FCFMargin': avg(fcf_margins), 'EpsGrowth': avg(eps_growths),
        'PEG': avg(pegs), 'ROIC': avg(roics), 'NetDebtEBITDA': avg(net_debt_ebitdas),
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


def _grade(val, metric, sector='', industry=''):
    if val is None:
        return '#95a5a6', 'N/A', True

    if metric == 'ROE':
        g, ok = _get_thresh('ROE', sector, industry)
        if val >= g:  return '#27ae60', f'우수 ({val}% ≥ {g}%)', False
        if val >= ok: return '#f39c12', f'보통 ({val}% / {ok}~{g}%)', False
        if val >= 0:  return '#e74c3c', f'위험 ({val}% < {ok}%)', True
        return            '#c0392b', f'위험 (음수 {val}%)', True

    if metric == 'PER':
        thresh = _get_thresh('PER', sector, industry)
        if thresh is None:
            return '#95a5a6', f'해당없음 ({val}) — 성장주 PER 비교 부적합', False
        g, ok = thresh
        if val <= 0:  return '#c0392b', f'위험 (음수/비정상)', True
        if val <= g:  return '#27ae60', f'저평가 ({val} ≤ {g})', False
        if val <= ok: return '#f39c12', f'보통 ({val} / {g}~{ok})', False
        return            '#e74c3c', f'고평가 ({val} > {ok})', True

    if metric == 'PSR':
        g, ok = _get_thresh('PSR', sector, industry)
        if val <= g:  return '#27ae60', f'저평가 ({val} ≤ {g})', False
        if val <= ok: return '#f39c12', f'보통 ({val} / {g}~{ok})', False
        return            '#e74c3c', f'고평가 ({val} > {ok})', True

    if metric == 'PBR':
        g, ok = _get_thresh('PBR', sector, industry)
        if val <= g:  return '#27ae60', f'저평가 ({val} ≤ {g})', False
        if val <= ok: return '#f39c12', f'보통 ({val} / {g}~{ok})', False
        return            '#e74c3c', f'고평가 ({val} > {ok})', True

    if metric == 'RevGrowth':
        if val >= 20:  return '#27ae60', f'고성장 ({val}% ≥ 20%)', False
        if val >= 5:   return '#f39c12', f'보통 ({val}% / 5~20%)', False
        if val >= 0:   return '#e74c3c', f'저성장 ({val}% < 5%)', True
        return             '#c0392b', f'역성장 ({val}%)', True

    if metric == 'DE':
        if val <= 1.0: return '#27ae60', f'안전 ({val} ≤ 1.0)', False
        if val <= 2.0: return '#f39c12', f'보통 ({val} / 1~2)', False
        return             '#e74c3c', f'위험 ({val} > 2.0)', True

    if metric == 'EV_EBITDA':
        if val <= 0:   return '#c0392b', '적자/비정상', True
        if val <= 12:  return '#27ae60', f'저평가 ({val} ≤ 12)', False
        if val <= 20:  return '#f39c12', f'보통 ({val} / 12~20)', False
        return             '#e74c3c', f'고평가 ({val} > 20)', True

    if metric == 'GrossM':
        if val >= 50:  return '#27ae60', f'우수 ({val}% ≥ 50%)', False
        if val >= 30:  return '#f39c12', f'보통 ({val}% / 30~50%)', False
        if val >= 0:   return '#e74c3c', f'낮음 ({val}% < 30%)', True
        return             '#c0392b', f'마이너스 ({val}%)', True

    if metric in ('OpM', 'NetM', 'FCFMargin'):
        if val >= 20:  return '#27ae60', f'우수 ({val}% ≥ 20%)', False
        if val >= 10:  return '#f39c12', f'보통 ({val}% / 10~20%)', False
        if val >= 0:   return '#e74c3c', f'낮음 ({val}% < 10%)', True
        return             '#c0392b', f'적자 ({val}%)', True

    if metric == 'FCFYield':
        if val >= 5:   return '#27ae60', f'매력적 ({val}% ≥ 5%)', False
        if val >= 2:   return '#f39c12', f'보통 ({val}% / 2~5%)', False
        if val >= 0:   return '#e74c3c', f'낮음 ({val}% < 2%)', True
        return             '#c0392b', f'현금소진 ({val}%)', True

    if metric == 'Beta':
        if val <= 0.8: return '#27ae60', f'방어적 ({val} ≤ 0.8)', False
        if val <= 1.2: return '#f39c12', f'시장수준 ({val} / 0.8~1.2)', False
        return             '#e67e22', f'공격적 ({val} > 1.2)', False

    if metric == 'CurrRatio':
        if val >= 2:   return '#27ae60', f'우수 ({val} ≥ 2.0)', False
        if val >= 1:   return '#f39c12', f'보통 ({val} / 1~2)', False
        return             '#e74c3c', f'위험 ({val} < 1.0)', True

    if metric == 'EpsGrowth':
        if val >= 20:  return '#27ae60', f'고성장 ({val}% ≥ 20%)', False
        if val >= 5:   return '#f39c12', f'보통 ({val}% / 5~20%)', False
        if val >= 0:   return '#e74c3c', f'저성장 ({val}% < 5%)', True
        return             '#c0392b', f'역성장 ({val}%)', True

    if metric == 'PEG':
        if val <= 0:   return '#c0392b', f'비정상 (PEG ≤ 0)', True
        if val <= 1:   return '#27ae60', f'저평가 ({val} ≤ 1.0) — 성장 대비 싸다', False
        if val <= 2:   return '#f39c12', f'적정 ({val} / 1.0~2.0)', False
        return             '#e74c3c', f'고평가 ({val} > 2.0)', True

    if metric == 'ROIC':
        if val >= 15:  return '#27ae60', f'우수 ({val}% ≥ 15%)', False
        if val >= 10:  return '#f39c12', f'보통 ({val}% / 10~15%)', False
        if val >= 0:   return '#e74c3c', f'낮음 ({val}% < 10%)', True
        return             '#c0392b', f'적자 (ROIC {val}%)', True

    if metric == 'NetDebtEBITDA':
        if val < 0:    return '#27ae60', f'순현금 ({val}x) — 무부채', False
        if val <= 2:   return '#27ae60', f'안전 ({val}x ≤ 2)', False
        if val <= 4:   return '#f39c12', f'보통 ({val}x / 2~4)', False
        return             '#e74c3c', f'위험 ({val}x > 4)', True

    return '#95a5a6', str(val), False


# ── 메인 함수 ────────────────────────────────────────────────────────────────

def analyze_financials(ticker, info):
    roe        = _safe(info, 'returnOnEquity', mult=100)
    per        = _safe(info, 'trailingPE')
    psr        = _safe(info, 'priceToSalesTrailing12Months')
    pbr        = _safe(info, 'priceToBook')
    ev_ebitda  = _safe(info, 'enterpriseToEbitda')
    rev_growth = _safe(info, 'revenueGrowth',  mult=100)
    eps_growth = _safe(info, 'earningsGrowth', mult=100)
    gross_m    = _safe(info, 'grossMargins',    mult=100)
    op_m       = _safe(info, 'operatingMargins',mult=100)
    net_m      = _safe(info, 'profitMargins',   mult=100)
    beta       = _safe(info, 'beta')
    curr_ratio = _safe(info, 'currentRatio')
    raw_de     = _safe(info, 'debtToEquity')
    de         = round(raw_de / 100, 2) if raw_de is not None else None
    fcf        = info.get('freeCashflow')
    mcap_v     = info.get('marketCap')
    rev_v      = info.get('totalRevenue')
    fcf_yield  = round(fcf / mcap_v * 100, 2) if (fcf and mcap_v and mcap_v > 0) else None
    fcf_margin = round(fcf / rev_v  * 100, 2) if (fcf and rev_v  and rev_v  > 0) else None

    # PEG: PER ÷ EPS성장률 (성장 대비 가격 적정성)
    peg = round(per / eps_growth, 2) if (per and eps_growth and eps_growth > 0 and per > 0) else None

    # ROIC: NOPAT ÷ 투하자본 (자본 효율성 — 부채 포함 관점)
    ebit_v     = info.get('ebit')
    tax_rate_v = info.get('effectiveTaxRate') or 0.21
    eq_v       = info.get('totalStockholderEquity') or 0
    total_debt_v = info.get('totalDebt') or 0
    total_cash_v = info.get('totalCash') or 0
    invested_cap = eq_v + total_debt_v - total_cash_v
    roic = round(ebit_v * (1 - tax_rate_v) / invested_cap * 100, 2) if (ebit_v is not None and invested_cap > 0) else None

    # 순부채/EBITDA: 레버리지 질적 지표
    ebitda_v = info.get('ebitda')
    net_debt_ebitda = round((total_debt_v - total_cash_v) / ebitda_v, 2) if (ebitda_v and ebitda_v > 0) else None

    # (label, key, val, desc)  |  key='__H__' → 섹션 헤더, val=bg색, desc=제목
    metrics = [
        ('__H__', '__H__', '#2c3e50',  '📊 밸류에이션'),
        ('PER',               'PER',          per,            '주가수익비율 — 낮을수록 저평가'),
        ('PEG',               'PEG',          peg,            'PER ÷ EPS성장률 — 1 이하=성장 대비 저평가, 2 초과=고평가'),
        ('PSR',               'PSR',          psr,            '주가매출비율 — 낮을수록 저평가'),
        ('PBR',               'PBR',          pbr,            '주가순자산비율 — 낮을수록 저평가'),
        ('EV / EBITDA',       'EV_EBITDA',    ev_ebitda,      '부채 포함 기업가치 ÷ EBITDA — 자본구조 무관 밸류에이션'),
        ('__H__', '__H__', '#27ae60',  '💰 수익성 & 현금창출'),
        ('ROE (%)',            'ROE',          roe,            '자기자본이익률 — 높을수록 자본 효율 우수'),
        ('ROIC (%)',           'ROIC',         roic,           'NOPAT ÷ 투하자본 — 부채 포함 자본효율, 15%↑ 우수'),
        ('매출총이익률 (%)',    'GrossM',       gross_m,        '가격경쟁력 지표 — 원가 통제력'),
        ('영업이익률 (%)',      'OpM',          op_m,           '핵심 사업 수익성 — 비용 통제력'),
        ('순이익률 (%)',        'NetM',         net_m,          '최종 이익률 — 회계 조정 후 실질 수익'),
        ('FCF 마진 (%)',        'FCFMargin',    fcf_margin,     '잉여현금흐름 ÷ 매출 — 진짜 현금창출력'),
        ('FCF 수익률 (%)',      'FCFYield',     fcf_yield,      'FCF ÷ 시가총액 — 5%↑ 채권 대비 매력적'),
        ('__H__', '__H__', '#3498db',  '🚀 성장성'),
        ('매출 성장률 YoY (%)', 'RevGrowth',    rev_growth,     '전년 대비 매출 증가율 — 20%↑ 고성장주'),
        ('EPS 성장률 YoY (%)',  'EpsGrowth',    eps_growth,     '전년 대비 주당순이익 증가율'),
        ('__H__', '__H__', '#e67e22',  '🛡️ 재무 안정성 & 리스크'),
        ('부채비율 D/E',        'DE',           de,             '부채÷자본 — 낮을수록 안전 (고금리 환경 특히 중요)'),
        ('순부채/EBITDA',       'NetDebtEBITDA',net_debt_ebitda,'(총부채-현금)÷EBITDA — 2 이하=안전, 4 초과=위험'),
        ('베타 (β)',            'Beta',         beta,           '시장 대비 변동성 — 1.0=시장동행 / 1.5↑=고변동'),
        ('유동비율',            'CurrRatio',    curr_ratio,     '유동자산÷유동부채 — 1.0 미만 시 단기 유동성 위험'),
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

    real_metrics = [(l, k, v, d) for l, k, v, d in metrics if k != '__H__']
    avail = sum(1 for _, _, v, _ in real_metrics if v is not None)
    total = len(real_metrics)
    ratio = avail / total if total else 0

    if ratio >= 0.75:
        rc, rl = '#27ae60', f'🟢 데이터 충분 — {avail}/{total} 지표 정상, 신뢰도 높음'
    elif ratio >= 0.50:
        rc, rl = '#f39c12', f'🟡 데이터 주의 — {avail}/{total} 지표 정상, 일부 N/A'
    else:
        rc, rl = '#e74c3c', f'🔴 데이터 부족 — {avail}/{total}만 확인됨, 투자 판단 주의'

    has_peer = etf_avgs is not None
    colspan  = 5 if has_peer else 4

    rows = ''
    for label, key, val, desc in metrics:
        # 섹션 헤더 행
        if key == '__H__':
            rows += (f'<tr style="background:{val}">'
                     f'<td colspan="{colspan}" style="padding:7px 12px;color:white;'
                     f'font-weight:bold;font-size:12px;letter-spacing:0.5px">{desc}</td></tr>')
            continue

        color, status, _ = _grade(val, key, sector, industry)
        val_str = 'N/A' if val is None else str(val)

        if has_peer:
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
                <span style="background:{color};color:white;border-radius:5px;padding:3px 10px;
                             font-size:12px;font-weight:bold">{status}</span>
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
