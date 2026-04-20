# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="7d267061"
# # LangChain으로 만드는 나만의 '투자 분석 봇'
#
# ## 학습 목표
# - LangChain 프레임워크로 **LLM 모델 간편 전환** 체험
# - 29차시 재무제표 분석을 LangChain Agent로 재구현
# - Tool 기반 데이터 수집 + 대화형 분석 봇 구축
#
# ## 학습 내용
# 1. 왜 LangChain인가? (29차시와 비교)
# 2. LangChain 모델 초기화
# 3. 네이버 금융 크롤링 Tool 정의
# 4. Memory Agent로 대화형 분석 봇 구축
# 5. Interactive Chatbot 실행

# %% colab={"base_uri": "https://localhost:8080/"} id="ee0e0eab" outputId="e50c4578-ea46-4a3f-9a09-56480acf6ede"
# #!pip install -Uq langchain langchain-openai langchain-google-genai langgraph python-dotenv requests beautifulsoup4 lxml

# %% colab={"base_uri": "https://localhost:8080/", "height": 150} id="04371f04" outputId="ccb9a6fb-60dc-4365-cf8a-691668ad9313"
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from IPython.display import display, Markdown

# .env 파일 로드
load_dotenv()

# Colab에서 .env 파일 업로드
try:
    from google.colab import files
    print("[Colab 환경 감지]")
    print("=" * 60)
    print(".env 파일을 업로드해주세요.")
    print()
    uploaded = files.upload()
    load_dotenv('.env')
except ImportError:
    print("[로컬 환경]")
    print("=" * 60)
    print(".env 파일이 현재 디렉토리에 있어야 합니다.")

# %% colab={"base_uri": "https://localhost:8080/"} id="8e42359c" outputId="4196b6ff-46af-4821-cc7f-e5ccc997200e"
# API 키 확인
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

print(f"[Gemini API 키 로드 완료]: {GEMINI_API_KEY[:10]}...")

# %% [markdown] id="50592c49"
# ---
# ## 1. 왜 LangChain인가?
#
# ### LangChain 방식: 통일된 인터페이스
# ```python
# from langchain.chat_models import init_chat_model
#
# # 한 줄로 모델 전환!
# model = init_chat_model("gpt-4o-mini", model_provider="openai")
# # model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
#
# # 동일한 호출 방식
# result = model.invoke(messages)
# ```
#
# ### 비교 요약
# | 항목 | 29차시 (직접 호출) | 30차시 (LangChain) |
# |------|-------------------|-------------------|
# | 모델 전환 | 함수 분리 필요 | 변수 하나로 전환 |
# | 호출 방식 | API별 상이 | `invoke()` 통일 |
# | 도구 연동 | 직접 구현 | `@tool` 데코레이터 |
# | 대화 기록 | 직접 관리 | Checkpointer |

# %% [markdown] id="d87f2f10"
# ---
# ## 2. LangChain 모델 초기화
#
# `init_chat_model()`로 다양한 LLM을 동일한 인터페이스로 사용합니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="3f79cc9e" outputId="39f56441-4061-49a1-d16d-61011de15593"
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
# model = init_chat_model("gpt-5-mini", model_provider="openai")
model

# %% [markdown] id="4132ed84"
# ---
# ## 3. 네이버 금융 크롤링 Tool 정의
#
# 29차시에서 사용한 크롤링 함수를 LangChain `@tool`로 래핑합니다.

# %% id="938e81c2"
# 공통 헤더 설정 (29차시와 동일)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_company_name(stock_code):
    """종목코드로 회사명 조회 (29차시 코드 재사용)"""
    url = f"https://finance.naver.com/item/main.nhn?code={stock_code}"
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        name_tag = soup.select_one('div.wrap_company h2 a')
        if name_tag:
            return name_tag.text.strip()
        return stock_code
    except Exception:
        return stock_code

def get_financial_summary(stock_code):
    """네이버 금융에서 재무 요약 정보 수집 (29차시 코드 재사용)"""
    url = f"https://finance.naver.com/item/main.nhn?code={stock_code}"
    try:
        tables = pd.read_html(url, encoding='euc-kr')
        financial_table = None
        for table in tables:
            table_str = str(table.columns) + str(table.values)
            if '매출액' in table_str or '영업이익' in table_str:
                financial_table = table
                break
        if financial_table is None:
            return None
        company_name = get_company_name(stock_code)
        return {
            'company_name': company_name,
            'stock_code': stock_code,
            'table': financial_table
        }
    except Exception as e:
        return {'error': str(e)}

def format_for_llm(raw_data):
    """수집한 데이터를 LLM 입력용 텍스트로 변환 (29차시 코드 재사용)"""
    if raw_data is None:
        return "데이터가 없습니다."
    if 'error' in raw_data:
        return f"데이터 수집 오류: {raw_data['error']}"

    lines = []
    lines.append(f"회사명: {raw_data['company_name']}")
    lines.append(f"종목코드: {raw_data['stock_code']}")
    lines.append(f"데이터 출처: 네이버 금융")
    lines.append(f"수집 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n[재무제표 데이터]")
    lines.append(raw_data['table'].to_string())
    return "\n".join(lines)

# %% id="6b0c250a"
# 네이버 금융 크롤링을 LangChain Tool로 정의
from langchain.tools import tool

@tool
def fetch_financial_data(stock_code: str) -> str:
    """
    종목코드로 네이버 금융에서 재무제표를 가져옵니다.
    예시: fetch_financial_data("005930") -> 삼성전자 재무제표
    주요 종목코드: 005930(삼성전자), 000660(SK하이닉스), 035420(NAVER)
    """
    print(f"  [Tool 호출] 종목코드 {stock_code} 재무제표 수집 중...")
    data = get_financial_summary(stock_code)
    return format_for_llm(data)

# %% colab={"base_uri": "https://localhost:8080/"} id="e26b567e" outputId="2b64e152-1dbb-45bb-a017-93bc22d5ce87"
# Tool 테스트
print("[네이버 금융 크롤링 Tool 테스트]")
print("=" * 60)
result = fetch_financial_data.invoke("005930")
print(result[:500] + "...")

# %% [markdown] id="73729c10"
# ---
# ## 4. Memory Agent로 대화형 분석 봇 구축
#
# 처음부터 **메모리(Checkpointer)** 가 포함된 Agent를 생성합니다.
# 이전 대화 내용을 기억하여 맥락을 유지합니다.

# %% colab={"base_uri": "https://localhost:8080/", "height": 266} id="029c48aa" outputId="abd98dd2-e7f4-42e3-fc46-518c8bf802e2"
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 사용할 도구 목록
tools = [fetch_financial_data]

# 시스템 프롬프트 (29차시 프롬프트 활용)
system_prompt = """당신은 15년 경력의 CFA 자격증을 보유한 증권 애널리스트입니다.

사용자가 기업 분석을 요청하면:
1. fetch_financial_data 도구로 재무제표를 수집합니다
2. 수집된 데이터를 기반으로 분석합니다

분석 시 다음 항목을 포함하세요:
- 수익성: 매출액, 영업이익, 당기순이익 추이
- 안정성: 부채비율, 유동비율 등
- 투자 포인트: 강점과 리스크

이전 대화 내용을 기억하고, 맥락에 맞게 답변합니다.
한국어로 답변합니다."""

# 메모리 (Checkpointer)
checkpointer = InMemorySaver()

# Agent 생성
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

agent

# %% [markdown] id="e0c1fa3e"
# ---
# ## 5. Interactive Chatbot 실행
#
# Agent에게 질문을 입력하고, 대화형으로 재무 분석을 수행합니다.
#
# ### 예시 질문
# 1. **기업 분석 요청**
#    - "삼성전자(005930)의 재무제표를 분석해줘"
#    - "SK하이닉스(000660)의 성장성을 분석해줘"
#    - "NAVER(035420)의 수익성을 분석해줘"
#
# 2. **추가 질문 (맥락 유지)**
#    - "영업이익률은 어때?"
#    - "투자 리스크는 뭐야?"
#    - "작년과 비교하면 어때?"
#
# 3. **비교 분석**
#    - "삼성전자와 SK하이닉스를 비교해줘"
#    - "두 회사 중 어디가 더 투자 가치가 있어?"
#
# ### 종료 방법
# - `exit` 또는 `quit` 입력

# %% colab={"base_uri": "https://localhost:8080/"} id="6bA1woRhxJ-B" outputId="6613e4fa-3ffa-42bf-9f75-234956ed763b"
print("[투자 분석 봇 시작]")
print("=" * 60)
print("질문을 입력하세요. 종료하려면 'exit' 입력")
print("예시: '삼성전자(005930)의 재무제표를 분석해줘'")
print("=" * 60)

# 세션 ID로 대화 맥락 유지
config = {"configurable": {"thread_id": "finance_session_001"}}

while True:
    # 사용자 입력
    user_input = input("\n사용자: ")

    # 종료 조건
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("\n[분석 봇 종료] 감사합니다!")
        break

    # 빈 입력 무시
    if not user_input.strip():
        continue

    # Agent 호출
    print("\n분석봇: ", end="")
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        )
        answer = result["messages"][-1].content
        print(answer)
    except Exception as e:
        print(f"오류 발생: {e}")

# %% id="53d67b42"
