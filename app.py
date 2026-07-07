from io import StringIO

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

URL = "https://finance.naver.com/sise/sise_market_sum.naver"


def fetch_market_sum_page(url: str = URL) -> str:
    """네이버 금융 시가총액 페이지 HTML 가져오기"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.naver.com/",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = "euc-kr"

    if response.status_code != 200:
        raise RuntimeError(f"페이지 요청 실패: {response.status_code}")

    return response.text


def parse_market_sum_table(html: str) -> pd.DataFrame:
    """네이버 시가총액 페이지에서 종목 테이블 추출"""
    tables = pd.read_html(StringIO(html))

    df = tables[1]
    df = df.dropna(how="all")

    if "토론실" in df.columns:
        df = df.drop(columns=["토론실"])

    df = df[df["종목명"].notna()]
    df = df.reset_index(drop=True)

    return df


def clean_market_data(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """AI에게 넘기기 좋게 데이터 정리"""
    df = df.copy()
    df = df.head(top_n)

    cols = [
        "N",
        "종목명",
        "현재가",
        "전일비",
        "등락률",
        "액면가",
        "시가총액",
        "상장주식수",
        "외국인비율",
        "거래량",
        "PER",
        "ROE",
    ]

    available_cols = [col for col in cols if col in df.columns]
    df = df[available_cols]

    return df


def dataframe_to_text(df: pd.DataFrame) -> str:
    """DataFrame을 LLM에 넣기 좋은 텍스트로 변환"""
    return df.to_string(index=False)


@st.cache_resource
def get_summary_chain():
    prompt = ChatPromptTemplate.from_template("""
너는 증권사 MTS에 들어가는 '오늘의 국내증시 요약' 작성자다.

아래 데이터는 네이버 금융 시가총액 상위 종목 데이터다.

[시가총액 상위 종목 데이터]
{market_data}

[작성 조건]
- 한국어로 작성
- 투자 권유처럼 쓰지 말 것
- 데이터에 없는 내용은 추측하지 말 것
- 종목 추천 금지
- 과장된 표현 금지
- MTS 화면에 들어갈 수 있도록 간결하게 작성
- 상승/하락 종목, 대형주 흐름, 업종 분위기를 중심으로 요약

[출력 형식]
1. 한 줄 요약:
2. 대형주 흐름:
3. 상승/하락 특징:
4. 외국인/거래량 특징:
5. 투자자가 볼 포인트:
6. 유의사항:
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    return prompt | llm | StrOutputParser()


def summarize_market_with_ai(market_text: str) -> str:
    """LangChain으로 AI 증시 요약 생성"""
    chain = get_summary_chain()
    return chain.invoke({"market_data": market_text})


st.set_page_config(page_title="오늘의 국내증시 요약", page_icon="📈")
st.title("📈 오늘의 국내증시 요약")
st.write("네이버 금융 시가총액 상위 종목 데이터를 가져와 AI가 요약해줍니다.")

top_n = st.slider("상위 몇 개 종목을 분석할까요?", min_value=10, max_value=50, value=30, step=5)

if st.button("증시 데이터 가져오기 & 요약하기", type="primary"):
    with st.spinner("네이버 금융에서 데이터를 가져오는 중..."):
        html = fetch_market_sum_page()
        df_raw = parse_market_sum_table(html)
        df_clean = clean_market_data(df_raw, top_n=top_n)

    st.subheader("수집 데이터")
    st.dataframe(df_clean, width="stretch")

    market_text = dataframe_to_text(df_clean)

    with st.spinner("AI가 증시를 요약하는 중..."):
        summary = summarize_market_with_ai(market_text)

    st.subheader("AI 증시 요약")
    st.markdown(summary)
