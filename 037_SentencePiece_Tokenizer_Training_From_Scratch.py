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

# %% [markdown] id="whjogPl1KL4-"
# # 037. 토크나이저 처음부터 학습 시키기

# %% colab={"base_uri": "https://localhost:8080/"} id="fvHF5gEXjYcd" outputId="69eeef5f-3b9c-47c9-b476-909c0735114f"
# KoNLPy(한국어 형태소 분석기 패키지) 설치
# !pip install -q KoNLPy

# SentencePiece(서브워드 토크나이저 도구)를 최신 버전으로 업그레이드 및 설치
# !pip install -U -q sentencepiece

# %% id="pUL4Bew4jYcg"
sentences_E = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'I was born in Korea and graduaged University in USA.',
]

sentences_K = [
    "코로나가 심하다",
    "코비드-19가 심하다",
    '아버지가방에들어가신다',
    '아버지가 방에 들어가신다',
    '너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다'
]

# %% [markdown] id="On8u7bpSjYci"
# # 1. Keras 기본 Tokenizer - rule-based
# - 공백 또는 구둣점으로 분리  
# - 영어 단어별로 띄어쓰기가 철저히 지켜지는 언어

# %% colab={"base_uri": "https://localhost:8080/"} id="xfOfMf9bjYcj" outputId="d73ba684-55ab-4a16-fb1b-43e6b70bae93"
from tensorflow.keras.preprocessing.text import Tokenizer

# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 주어진 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(sentences_E)

# 구축된 단어 인덱스 사전 가져오기
word_index = tokenizer.word_index

# 단어 인덱스 사전 출력
print(word_index)

# %% [markdown] id="0TJA8onMkTiy"
# Keras의 rule base tokenizer로 한글을 tokenize

# %% colab={"base_uri": "https://localhost:8080/"} id="theDox2EjYcl" outputId="ee1b5197-3501-4a96-ffc0-5e2ed87fcb8c"
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 주어진 한글 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(sentences_K)

# 구축된 단어 인덱스 사전 가져오기
vocabulary_keras_korean = tokenizer.word_index

# 단어 인덱스 사전 출력
print(vocabulary_keras_korean)

# %% [markdown] id="TkQ9u94VjYcl"
# # 2. 단어 사전 기반 한국어 tokenizer 사용

# %% colab={"base_uri": "https://localhost:8080/"} id="KQKMuFq7MmJ6" outputId="76a8046d-e007-4bf7-cff9-0262111af5da"
from konlpy.tag import Okt

# Okt 형태소 분석기 객체 생성
okt = Okt()

# 형태소 분석 결과를 저장할 리스트 초기화
temp_X = []

# 주어진 한글 문장 리스트의 각 문장에 대해 반복
for sent in sentences_K:
    # 문장을 형태소 분석하여 결과를 리스트에 추가
    temp_X.append(okt.morphs(sent))
    # 형태소 분석 결과 출력
    print(okt.morphs(sent))

# %% [markdown] id="MKy3rIq0kuLo"
# 사전 기반 tokenize 후 Keras tokenizer 로 vocabulary 생성

# %% colab={"base_uri": "https://localhost:8080/"} id="jIM1k4UgjYcn" outputId="09920c25-7305-4f18-b465-bf00123ddbf7"
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 형태소 분석된 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(temp_X)

# 구축된 단어 인덱스 사전 가져오기
vocabulary_okt_keras = tokenizer.word_index

# 단어 인덱스 사전 출력
print(vocabulary_okt_keras)

# %% [markdown] id="6FRpIxColOLv"
# 두 vocabulary 의 차이 비교

# %% colab={"base_uri": "https://localhost:8080/"} id="qUjQL_94lCWO" outputId="6f36a0af-7dff-4b9f-dd33-0d15041a753b"
print(vocabulary_keras_korean)
print(vocabulary_okt_keras)

# %% [markdown] id="MRtVftAuMmJ8"
# ### 단, Okt 사전에 미등록된 단어의 경우 정확한 tokenizing 이 안된다.

# %% colab={"base_uri": "https://localhost:8080/"} id="YUpB1dYWMmJ8" outputId="4b64b226-a1d1-419b-8319-6d08b990ede3"
# 주어진 문장을 형태소 분석하여 품사 태깅 수행
okt.pos('너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다')

# %% [markdown] id="6Cq9CJhJMmJ9"
# 예를 들어 `너무너무너무`와 `나카무라세이코`는 하나의 단어이지만, okt 사전에 등록되어 있지 않아 여러 개의 복합단어로 나뉘어집니다. 이러한 문제를 해결하기 위하여 형태소 분석기와 품사 판별기들은 사용자 사전 추가 기능을 제공합니다. 사용자 사전을 추가하여 모델의 vocabulary 를 풍부하게 만드는 것은 사용자의 몫입니다.
#
# 1. okt 공식 문서를 참고해서 사용사 사전을 추가.
# 2. okt를 패키징하고, konlpy에서 사용할 수 있도록 konlpy/java 경로에 jar 파일을 복사.
# 3. 기존에 참고하고 있던 okt.jar 대신 새로운 okt.jar를 사용하도록 설정.
# 4. konlpy 소스 경로를 import 해서 형태소 분석.

# %% [markdown] id="9NSFEmKRMmKB"
# # 3. Google SentencePiece Tokenizer
#
# - NAVER Movie rating data 를 이용한 sentencepiece tokenizer training

# %% colab={"base_uri": "https://localhost:8080/"} id="f-9k6x1ZMmKC" outputId="d4a3fc50-83b0-44c6-baa8-fb32039172df"
import tensorflow as tf
import pandas as pd
import sentencepiece as spm

DATA_TRAIN_PATH = tf.keras.utils.get_file("ratings_train.txt",
        "https://github.com/ironmanciti/infran_NLP/raw/main/data/naver_movie/ratings_train.txt")

# %% [markdown] id="Vn-WA_c6MmKC"
# - pandas.read_csv에서 quoting = 3으로 설정해주면 인용구(따옴표)를 무시

# %% colab={"base_uri": "https://localhost:8080/", "height": 224} id="5q_gXlchMmKD" outputId="ffe20c5d-18d2-46dc-ce03-70719fae5735"
train_data = pd.read_csv(DATA_TRAIN_PATH, sep='\t', quoting=3)

print(train_data.shape)
train_data.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="42U-cU9lMmKD" outputId="c9542528-f2ce-4506-f77a-fcab4903823c"
train_data.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/"} id="f3Bm3wxwMmKD" outputId="dc5b8a87-e64e-41ca-95d8-deb53edec34c"
train_data.dropna(inplace=True)

train_data.shape

# %% [markdown] id="4h4yXblIMmKE"
# ## 학습을 위해 text 를 따로 저장

# %% id="U8HVwXaQMmKE"
# 'nsmc.txt' 파일을 쓰기 모드로 열기 (UTF-8 인코딩 사용)
with open('./nsmc.txt', 'w', encoding='utf-8') as f:
    # 훈련 데이터의 'document' 열에 있는 각 문장에 대해 반복
    for line in train_data.document.values:
        try:
            # 문장을 파일에 쓰고 새로운 줄 추가
            f.write(line + '\n')
        except:
            # 쓰기 오류 발생 시 오류 메시지와 해당 문장 출력
            print("write error ---> ", line)

# %% colab={"base_uri": "https://localhost:8080/"} id="WML97jwWMmKE" outputId="4ca96fa4-34f0-4e44-d1b5-77f86e9a03c2"
#write 가 잘 되었는지 확인
with open('./nsmc.txt', 'r', encoding='utf-8') as f:
    nsmc_txt = f.read().split('\n')

print(len(nsmc_txt))
print(nsmc_txt[0])

# %% colab={"base_uri": "https://localhost:8080/"} id="ENQL0fs5MmKF" outputId="54d7ce94-57f6-4df5-8208-93ac739b12cd"
# 입력 파일 경로 설정
input_file = 'nsmc.txt'

# 어휘 사전의 최대 크기 설정
vocab_size = 30000

# 모델 파일의 접두사 설정
prefix = 'nsmc'

# 명령어 템플릿 정의
templates = '--input={} --model_prefix={} --vocab_size={}'

# 템플릿에 변수 값을 포맷하여 명령어 문자열 생성
cmd = templates.format(input_file, prefix, vocab_size)

# 생성된 명령어 출력
print(cmd)

# %% [markdown] id="0IcsoxuvMmKF"
# ### sentencepiece tokenizer training

# %% id="WaYRD7mjMmKF"
# SentencePieceTrainer를 사용하여 SentencePiece 모델 학습
spm.SentencePieceTrainer.Train(cmd)

# %% colab={"base_uri": "https://localhost:8080/"} id="8zvm9kadMmKG" outputId="946e5454-6f6c-40c3-aa36-edc86a7fbedc"
# SentencePieceProcessor 객체 생성
sp = spm.SentencePieceProcessor()

# 학습된 SentencePiece 모델 로드
sp.Load('{}.model'.format(prefix))

# %% colab={"base_uri": "https://localhost:8080/"} id="CBl78txHMmKG" outputId="b94b08f4-9b63-400c-f409-13b96a069f5e"
# 훈련 데이터의 'document' 열에 있는 첫 세 개의 문장에 대해 반복
for t in train_data.document.values[:3]:
    # 원본 문장 출력
    print(t)
    # 문장을 SentencePiece 모델을 사용하여 토큰화하여 출력
    print(sp.encode_as_pieces(t))
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환하여 출력
    print(sp.encode_as_ids(t), '\n')

# %% colab={"base_uri": "https://localhost:8080/"} id="VG8_6rToMmKG" outputId="ba00bad2-5703-496f-ca6a-0f7aeaafd85f"
# 한글 문장 리스트(sentences_K)에 있는 각 문장에 대해 반복
for line in sentences_K:
    # 문장을 SentencePiece 모델을 사용하여 토큰화
    pieces = sp.encode_as_pieces(line)
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환
    ids = sp.encode_as_ids(line)
    # 원본 문장 출력
    print(line)
    # 토큰화된 결과 출력
    print(pieces)
    # 인덱스 시퀀스 출력
    print(ids)
    # 각 문장 사이에 줄 바꿈 추가
    print()
