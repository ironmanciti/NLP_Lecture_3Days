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

# %% [markdown] id="659dbb67-e15f-4d2b-a927-c98ca7973fdb"
# # OpenAI의 tiktoken을 사용한 tokenizer 개념 이해
#
# - tiktoken은 OpenAI에서 개발한 빠르고 효율적인 BPE(Byte Pair Encoding) 기반 토크나이저입니다.  
# - GPT 모델들이 사용하는 것과 동일한 토크나이징 방식을 제공합니다.
#
# ```
#     "cl100k_base": "GPT-4, GPT-3.5-turbo, text-embedding-ada-002에서 사용",
#     "p50k_base": "GPT-3, Codex에서 사용",
#     "r50k_base": "GPT-3, GPT-2에서 사용"
# ```
#
# - tiktoken은 BPE(Byte Pair Encoding) 방식을 사용합니다. 가장 자주 등장하는 문자 쌍을 하나의 토큰으로 병합하는 방식입니다.

# %% id="d7df5a8e-64d8-4c42-b636-ae33c77e385b"
import tiktoken
import pandas as pd
import numpy as np

# 연습용 텍스트 데이터
sentences_E = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'I was born in Korea and graduated University in USA.',
]

sentences_K = [
    "코로나가 심하다",
    "코비드-19가 심하다",
    '아버지가방에들어가신다',
    '아버지가 방에 들어가신다',
    '너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다'
]

# %% [markdown] id="a22b7b5b-9aa5-4453-bdd0-8fd108f29f00"
# ### cl100k_base 인코더로 영어 텍스트 토크나이징

# %% colab={"base_uri": "https://localhost:8080/"} id="9ef73d1e-2da2-4df4-86e2-872b5f8a19cd" outputId="9829fa72-bd5a-4135-f977-3ce7ad227634"
# cl100k_base 인코더 사용 (GPT-5와 동일)
encoding = tiktoken.get_encoding("cl100k_base")

# 영어 문장 리스트 순회
for i, sentence in enumerate(sentences_E):
    print(f"\n문장 {i+1}: {sentence}")

    # 텍스트를 토큰으로 변환 (문장을 토큰 ID 리스트로 인코딩)
    tokens = encoding.encode(sentence)
    print(f"토큰 ID: {tokens}")

    # 토큰 ID를 다시 텍스트로 복원 (디코딩)
    decoded_text = encoding.decode(tokens)
    print(f"디코딩 결과: {decoded_text}")

    # 토큰 개수 출력
    token_count = len(tokens)
    print(f"토큰 개수: {token_count}")

# %% [markdown] id="d2776c78-cfd3-48b1-bf95-b631b1d7df1a"
# ### 한글 텍스트 토크나이징
# - 한글은 영어와 달리 띄어쓰기가 없어도 의미가 통하는 언어입니다.  

# %% colab={"base_uri": "https://localhost:8080/"} id="3e10ae52-bc76-4ed0-beee-172576692eb7" outputId="bcc44a45-1668-4727-eba7-97aba173fd79"
for i, sentence in enumerate(sentences_K):
    print(f"\n한글 문장 {i+1}: {sentence}")

    # 텍스트를 토큰으로 변환
    tokens = encoding.encode(sentence)
    print(f"토큰 ID: {tokens}")

    # 토큰을 다시 텍스트로 변환
    decoded_text = encoding.decode(tokens)
    print(f"디코딩 결과: {decoded_text}")

    # 토큰 개수
    token_count = len(tokens)
    print(f"토큰 개수: {token_count}")

# %% id="f7ace1b3-cc45-49c0-82b5-e897131e6c4c"
