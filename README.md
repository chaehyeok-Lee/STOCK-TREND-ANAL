# 📈 미국 주식 종합 분석 툴

미국 주식을 **재무 지표 → 기술적 차트 → 뉴스·거시경제** 순으로 자동 분석하는 퀀트 투자 도구입니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `main.ipynb` | 종목 티커 입력 → 3단계 종합 분석 실행 |
| `step1_financials.py` | STEP 1 — 재무 지표 분석 (ROE / PER / PSR / PBR / 매출성장률 / 부채비율) |
| `step2_charts.py` | STEP 2 — 기술적 차트 분석 (1y / 3y / 5y) |
| `step3_news.py` | STEP 3 — 뉴스 · 배당 · 거시경제 · 시나리오 분석 |
| `quant_run.py` | 섹터별 저평가 종목 스크리닝 (TOP 10 도출) |
| `quant_test.ipynb` | 나스닥 상위 50개 종목 퀀트 필터 테스트 |

---

## 주요 기능

### STEP 1 — 재무 지표
- ROE, PER, PSR, PBR, 매출 성장률(YoY), 부채비율 자동 계산
- 지표별 색상 평가 (🟢 우수 / 🟡 보통 / 🔴 위험)
- 데이터 신뢰도 등급 자동 표시

### STEP 2 — 기술적 차트
- 1년(캔들) / 3년 / 5년(라인) 차트 3개 자동 생성
- 50 / 100 / 150 / 200일 이동평균선
- MACD + 다이버전스 히스토그램
- RSI (과매수/과매도 구간 표시)
- S&P 500 정규화 비교선
- **이동평균선 기울기 연장 + 교점 가격·날짜 자동 표시**

### STEP 3 — 뉴스 & 시나리오
- 다음 실적 발표일, 월가 투자 의견, 목표주가 업사이드 %
- Forward/Trailing PER, 공매도 비율, 52주 가격 위치
- 배당금 이력 및 배당 수익률
- **글로벌 금리 실시간 표시** (Fed 기준금리 / 미국 10Y 국채 / VIX / 달러 인덱스)
- 향후 60일 거시경제 일정 (FOMC / CPI / NFP)
- 섹터별 리스크 요인 자동 매핑
- 최근 뉴스 6개 (원문 링크 포함)
- Bull / Bear 시나리오 자동 생성

### quant_run.py — 섹터 스크리닝
- 반도체 / 소프트웨어 / 헬스케어 / 빅테크 / 에너지 / 금융 / 나스닥50 선택
- 방법 A: 섹터 내 **상대 퍼센타일** 점수법 TOP 10
- 방법 B: 가치투자 **절대 기준** 구간 점수법 TOP 10

---

## 사용법

### 종합 분석 (main.ipynb)
```
1. main.ipynb 실행
2. 티커 입력 (예: AAPL, NVDA, TSLA)
3. STEP 1~3 자동 출력
```

### 섹터 스크리닝 (quant_run.py)
```bash
python quant_run.py
# 섹터 번호 입력 → TOP 10 종목 자동 출력
```

---

## 설치

```bash
pip install yfinance pandas numpy matplotlib FinanceDataReader
```

---

## API 호출 구조 (종목당 5회 최소화)

| 호출 | 데이터 | 용도 |
|------|--------|------|
| `stock.info` | 재무 전체 | STEP 1·3 공용 |
| `stock.history(5y)` | 주가 5년치 | STEP 2 슬라이싱 재사용 |
| `stock.news` | 최근 뉴스 | STEP 3 |
| `stock.dividends` | 배당 이력 | STEP 3 |
| `stock.earnings_dates` | EPS 분기 이력 | STEP 3 |

S&P 500 기준선은 세션 시작 시 1회만 로드 후 재사용.

---

> 본 도구는 정보 제공 목적이며 투자 권유가 아닙니다.
> 데이터 출처: [yfinance](https://github.com/ranaroussi/yfinance), [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)
