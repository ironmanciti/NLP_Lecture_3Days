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

# %% [markdown] id="MQi7NaBDRwKj"
# # 300_HuggingFace_QuickStart_pipeline
#
# 파이프라인은 추론을 위해 모델을 사용하는 훌륭하고 쉬운 방법입니다.
#
# 이러한 파이프라인은 라이브러리에서 대부분의 복잡한 코드를 추상화하는 개체로, Named Entity Recognition, Masked Language Modeling, 감정 분석, Feature Extraction 및 Question Answering.을 비롯한 여러 task 전용의 간단한 API를 제공합니다.

# %%

# %%

# %% [markdown] id="1n42KmQdePXA"
# 🤗 Transformers 라이브러리 기능을 간단히 살펴보겠습니다.  
#
# 라이브러리는 **텍스트의 감정 분석과 같은 자연어 이해(NLU)** 및 **새 텍스트로 프롬프트를 완성하거나 다른 언어로 번역하는 것과 같은 자연어 생성(NLG)** 작업을 위해 사전 훈련된 모델을 다운로드합니다.
#
# 먼저 pipeline API를 쉽게 활용하여 추론에서 **사전 훈련된 모델**을 빠르게 사용하는 방법을 살펴보겠습니다. 그런 다음, 라이브러리가 어떻게 이러한 모델에 대한 액세스를 제공하고 **데이터를 사전 처리**하는 데 도움이 되는지 확인 할 것입니다.

# %% [markdown] id="U9xDJ3f0ePXB"
# ## pipeline 으로 작업 시작하기

# %% [markdown] id="74yhxzJzePXD"
# - 주어진 task 에서 사전 훈련된 모델을 사용하는 가장 쉬운 방법은  `pipeline`을 사용하는 것입니다.
#
# 🤗 Transformers 라이브러리는 기본적으로 다음 task를 제공합니다.
#
# - **기계 번역(Translation)**: 다른 언어로 된 텍스트를 번역합니다.  
# - **감정 분석(Text Classification)**: 텍스트는 긍정적인가 부정적인가?
# - **텍스트 생성(Text Generation)**: 프롬프트를 제공하면 모델이 다음을 생성합니다.
# - **이름 개체 인식(NER)**: 입력 문장에서 각 단어를 나타내는 개체(사람, 장소, 등.)
# - **질문 답변(Question Answering)**: 모델에 일부 컨텍스트와 질문을 제공하고 컨텍스트에서 답변을 추출합니다.
# - **마스킹된 텍스트 채우기(Fill-Mask)**: 마스킹된 단어가 있는 텍스트(예: `[MASK]`로 대체)가 주어지면 공백을 채웁니다.
# - **요약(Summarization)**: 긴 텍스트의 요약을 생성합니다.
# - **특징 추출(Feature Extraction)**: 텍스트의 텐서 표현을 반환합니다.
# - **Zero-Shot 분류(Zero-Shot Classification)**
#
#
# ### pretrained models : https://huggingface.co/models

# %% [markdown] id="ac3qDPYA6Xmk"
# ## 기계 번역
#
# - korean pretrained model : https://huggingface.co/Helsinki-NLP/opus-mt-ko-en  
#
# - Helsinki-NLP : University of Helsinki 에서 작성한 다양한 언어 모델 그룹

# %%
# HuggingFace Transformers의 번역 파이프라인 생성
# 한국어(Korean) → 영어(English) 번역 모델 사용

# %% [markdown] id="pcvvG9876Xmp"
# ## 한국어 감정분석
#
# - NSMC(Naver Sentiment Movie Corpus) 로 미세 조정된 BERT 다국어 basecase 모델 : https://huggingface.co/sangrimlee/bert-base-multilingual-cased-nsmc

# %%
# 한국어 감성 분석 파이프라인 생성
# 'sangrimlee/bert-base-multilingual-cased-nsmc' 모델은
# 네이버 영화 리뷰 데이터셋(NSMC)으로 학습된 다국어 BERT 기반 감성 분류 모델

# %%

# %%

# %% [markdown] id="qVRSJKEv5zb4"
# - 자동 별점 부여

# %%
# 다국어 감성 분석 모델 이름 지정
# 'nlptown/bert-base-multilingual-uncased-sentiment'는 영어뿐만 아니라 한국어 등 여러 언어의 감성 분석이 가능한 BERT 기반 모델
# 감성 분석 파이프라인 생성 (한국어 문장도 지원됨)

# %% [markdown] id="wr-tP0HF4CmP"
# ## Zero Shot Pipeline - 처음 보는 문장의 category 분류

# %%
# 제로샷 분류(zero-shot classification) 파이프라인 생성
# 사전 정의된 레이블(label)에 대해 학습 없이 문장을 분류할 수 있음
# 입력 문장을 주어진 후보 레이블 중 어떤 범주로 분류할지 예측

# %% [markdown] id="jzChBWjG6Xms"
# ### Zero Shot Pipeline - 다국어 version

# %%

# %% [markdown] id="Q9MuZlOpvh5Z"
# ### 개체명 인식

# %%
# 개체명 인식(NER: Named Entity Recognition) 파이프라인 생성
# grouped_entities=True: 같은 개체로 인식된 연속된 토큰들을 하나로 묶어서 반환
# 입력 문장에서 사람 이름, 기관명, 위치 등 고유 명사를 인식
# 결과 출력

# %% [markdown] id="piQOP7kL6Xmu"
# ### 한글 Text 생성
# HyperCLOVAX‑SEED‑Text‑Instruct‑0.5B는 지시문 기반 텍스트-투-텍스트 모델로, 한국어 언어 및 문화 이해에 뛰어난 성능을 보입니다. 유사한 규모의 외부 경쟁 모델과 비교했을 때 수학적 성능이 향상되고, 한국어 능력이 크게 개선되었습니다. 이 모델은 HyperCLOVAX 시리즈에서 현재 출시된 모델 중 가장 작은 모델이며, 에지 디바이스와 같은 리소스가 제한된 환경에 적합한 경량 솔루션입니다. 최대 4K 토큰의 컨텍스트 길이를 지원하며, 다양한 작업에 적용 가능한 다목적 소형 모델입니다.

# %%
def generate_response(system_content, user_content, max_length=1024, repetition_penalty=1.2):
    # 필요시 <|endofturn|>, <|stop|> 등에서 자르기


# %%

# %% [markdown] id="FvfXYfmKd5xY"
# ### 사전 학습된 모델을 이용한 챗봇
#

# %%
# 챗봇과 대화하는 함수 정의 (간단한 버전)
def chat_with_bot():
        # 사용자로부터 입력 받기
        # 종료 조건: 사용자가 quit, exit, bye 입력 시 대화 종료
            # generate_response 함수를 사용하여 응답 생성
            # regex로 "assistant" 이후의 내용만 추출하고 정리
            # 사용자 입력이 그대로 출력된 부분 제거
# 대화 시작

# %%
