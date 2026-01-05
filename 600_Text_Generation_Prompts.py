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

# %% [markdown] id="f609bd0c-79ec-4ce3-bfb0-4d17006a51e6"
# # Text 생성 및 Prompt 예제

# %% colab={"base_uri": "https://localhost:8080/"} id="989f8b65-992a-4c1c-b196-aa19cfff4e95" outputId="beaec10e-9781-4b35-ce43-78029583ffed"
import os
import openai

from dotenv import load_dotenv
load_dotenv() # read local .env file

# %% colab={"base_uri": "https://localhost:8080/"} id="9a83b1e3-b3f3-42f4-a73f-7bc62e3eb87a" outputId="8cc23beb-1ac6-4d89-f3d9-0edcda5ca6ad"
from openai import OpenAI
import google.generativeai as genai

client = OpenAI()

model_openai = "gpt-5-nano"
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

model_openai, model_gemini

# %% [markdown] id="2b00e72f-a786-472f-b849-27bd21ea6b7c"
# ## Responses API

# %% [markdown] id="6e00626d-ca14-411b-898f-856309a98b6d"
# ## 문법 수정
#
# SYSTEM : 당신은 문장을 받게 될 것이며, 당신의 임무는 그것을 표준 한국어로 변환하는 것입니다.  
# USER :   
# 안갔어 시장에 그녀는.

# %% colab={"base_uri": "https://localhost:8080/"} id="Bn1fNt8Yna3n" outputId="210c8065-549b-494e-fc17-ac2935721a78"
system_prompt = "당신은 문장을 받게 될 것이며, 당신의 임무는 그것을 표준 한국어로 변환하는 것입니다."
user_input = "안갔어 시장에 그녀는."

# OpenAI  API 호출 예시
response = client.responses.create(
    model=model_openai,
    instructions=system_prompt,
    input=user_input
)

# 응답 결과 출력
print(response.output_text)

# %% colab={"base_uri": "https://localhost:8080/"} id="43c10186-4a28-4ac1-ab16-82ab61d96fe3" outputId="7dfff0ff-999a-4866-8f21-fb171d4e9f5c"
# Gemini  API 호출 예시
prompt = f"""{system_prompt}

입력: {user_input}
출력:"""

response = model_gemini.generate_content(prompt)
# 응답 결과 출력
print(response.text)

# %% [markdown] id="54f4415f-60b6-4308-aaf1-9f46dc634445"
# ## 구조화되지 않은 데이터의 구문 분석
# SYSTEM : 구조화되지 않은 데이터가 제공되며 이를 CSV 형식으로 구문 분석하는 작업이 수행됩니다.  
# USER :   
# 최근 발견된 행성 구크럭스(Goocrux)에서는 많은 과일이 발견됐다. 그곳에서 자라는 네오스키즐이 있는데, 보라색이고 사탕 맛이 납니다. 회색 빛이 도는 파란색 과일이고 매우 시큼하며 레몬과 약간 비슷한 로헤클(loheckles)도 있습니다. 포유닛은 밝은 녹색을 띠며 단맛보다 풍미가 더 좋습니다. 네온 핑크색 맛과 솜사탕 같은 맛이 나는 루프노바도 많이 있습니다. 마지막으로 글로울(glowls)이라는 과일이 있는데, 이 과일은 신맛과 부식성이 있는 매우 신맛과 쓴맛이 나며 옅은 오렌지색을 띠고 있습니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="1a90e4b6-10c7-48bc-9da2-e617b32cef18" outputId="f83f2716-6f86-4eb0-ac8a-fa74868c91e0"
system_prompt = "구조화되지 않은 데이터가 제공되며 이를 CSV 형식으로 구문 분석하는 작업이 수행됩니다."
user_input = """
      최근 발견된 행성 구크럭스(Goocrux)에서는 많은 과일이 발견됐다. 그곳에서 자라는 네오스키즐이 있는데, 보라색이고 사탕 맛이 납니다.
      회색 빛이 도는 파란색 과일이고 매우 시큼하며 레몬과 약간 비슷한 로헤클(loheckles)도 있습니다. 포유닛은 밝은 녹색을 띠며 단맛보다 풍미가 더 좋습니다.
      네온 핑크색 맛과 솜사탕 같은 맛이 나는 루프노바도 많이 있습니다. 마지막으로 글로울(glowls)이라는 과일이 있는데,
      이 과일은 신맛과 부식성이 있는 매우 신맛과 쓴맛이 나며 옅은 오렌지색을 띠고 있습니다.
      이 데이터를 과일명, 색상, 맛으로 분석해 주세요.
      """

response = client.responses.create(
    model=model_openai,
    instructions=system_prompt,
    input=user_input
)

print(response.output_text)

# %% colab={"base_uri": "https://localhost:8080/"} id="8e85a03c-135c-4908-b24e-622a0c408a71" outputId="af1258b4-9539-40ab-c275-d4b88f960851"
prompt_csv = f"""{system_prompt}

입력:
{user_input}

CSV 형식으로 출력:"""

response = model_gemini.generate_content(prompt_csv)

print(response.text)

# %% [markdown] id="05e0d9c0-12f4-4ed8-aaf8-a704505fcb98"
# ## Keyword 추출

# %% id="c99d0449-9850-4e25-b779-2e1041a1577a"
system_prompt = "텍스트 블록이 제공되며, 당신의 임무는 텍스트 블록에서 키워드 목록을 추출하는 것입니다."

text = """
"블랙 온 블랙 도자기(Black-on-Black ware)는 뉴멕시코 북부의 푸에블로 원주민 도자기 예술가들이 개발한 20세기 및 21세기 도자기 전통입니다.
전통적인 환원 소성 블랙웨어는 푸에블로 예술가들에 의해 수세기 동안 만들어졌습니다.
지난 세기의 흑색 자기는 표면이 매끄럽고 선택적 버니싱이나 내화 슬립을 적용하여 디자인을 적용한 제품입니다.
또 다른 스타일은 디자인을 조각하거나 절개하고 융기된 부분을 선택적으로 연마하는 것입니다.
여러 세대에 걸쳐 Kha'po Owingeh와 P'ohwhóge Owingeh 푸에블로의 여러 가족은 여주인 도예가들로부터 전수받은 기술을 사용하여 검은 바탕에 검은 도자기를 만들어 왔습니다.
다른 푸에블로 출신의 예술가들도 검정색 바탕에 검정색 도자기를 제작했습니다. 몇몇 현대 예술가들은 조상의 도자기를 기리는 작품을 만들었습니다."
"""

# %% colab={"base_uri": "https://localhost:8080/"} id="fac052fa-6db9-4b98-a995-8fe8eb32ee60" outputId="fb164060-500a-4283-d5a4-47bf2681299c"
response = client.responses.create(
    model=model_openai,
    instructions=system_prompt,
    input=text
)

print(response.output_text)

# %% colab={"base_uri": "https://localhost:8080/"} id="6e9c970c-d280-4d5c-a4d8-bf9ab8ffe51a" outputId="d545a26c-1d12-4605-f321-8c57a2e65662"
prompt_keywords = f"""{system_prompt}

텍스트:
{text}

키워드 목록:"""

response = model_gemini.generate_content(prompt_keywords)

print(response.text)

# %% [markdown] id="a1934098-ba94-4cce-a64b-0a46566f67c2"
# ## 감성 분류기
# - 한개의 text 감성 분석

# %% colab={"base_uri": "https://localhost:8080/"} id="d9c959ce-c320-4ec9-8412-31152d6363ec" outputId="99c2c9eb-830b-4c41-a599-96fc37024147"
system_prompt = "당신은 텍스트를 입력 받게 될 것이고, 당신의 임무는 텍스트의 감정을 긍정적, 중립적, 부정적으로 분류하는 것입니다."
user_input = "나는 새로운 배트맨 영화가 좋습니다!"

response = client.responses.create(
    model=model_openai,
    instructions=system_prompt,
    input=user_input
)

print(response.output_text)

# %% colab={"base_uri": "https://localhost:8080/"} id="9ceeeec9-dcb0-4dde-8752-73efa1a82163" outputId="9dc68ea3-4ec3-45c4-e688-536cadd20af4"
prompt_sentiment = f"""{system_prompt}

입력: {user_input}
감정 분류:"""

response = model_gemini.generate_content(prompt_sentiment)

print(response.text)

# %% [markdown] id="47062606-7e80-4332-9bbc-f48681169d83"
# ## 실습: 위의 Prompt 내용을 수정해 가며 api 실행

# %% id="153ef2b1-9ba7-4da6-963a-77a52b07212c"
