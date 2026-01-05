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

# %% [markdown] id="bqRzQjEMbmvy"
# # 010. Vectorization of Statement (문장의 vector 화)
#
# - BOW (Bag of Words)
# - TF-IDF (Term Frequency - Inverse Document Frequency)  
# - Word Embedding - Keras word API 사용

# %% id="zgSqxnkIbmv0"
import pandas as pd

sentences =[
    '나는 내 개를 사랑해.',
    '나는 내 고양이를 사랑해.',
    '나는 내 개를 사랑하고 내 고양이도 사랑해.',
    '너는 내 개를 사랑하는구나!',
    '너는 내 개가 놀랍다고 생각해?'
]

# %% [markdown] id="wVoHTfTZbmv0"
# ## 1. Bag of Word (BOW)
#
# CountVectorizer는 Python의 scikit-learn 라이브러리에 포함된 클래스로, 텍스트 데이터의 토큰화(tokenization)와 단어 빈도 수를 기반으로 하는 피처 벡터(feature vector)를 생성하는 데 사용됩니다. 이 클래스는 자연어 처리(Natural Language Processing, NLP)와 텍스트 마이닝에서 널리 사용되며, 주요 기능은 다음과 같습니다:
#
# 토큰화(Tokenization): 문장이나 문서를 개별 단어나 표현으로 분할합니다.
#
# 단어 빈도 계산(Word Frequency Counting): 각 단어가 문서 내에서 나타나는 빈도를 계산합니다.
#
# 피처 벡터 생성(Feature Vector Creation): 각 문서를 단어의 빈도를 나타내는 벡터로 변환합니다. 이 벡터는 머신러닝 알고리즘에 입력으로 사용될 수 있습니다.
#
# 사전 구축(Vocabulary Building): 모든 문서에서 사용된 모든 단어의 사전을 만듭니다.
#
# CountVectorizer를 사용하면 텍스트 데이터를 수치적으로 분석할 수 있으며, 이는 감정 분석, 주제 모델링, 문서 분류와 같은 다양한 NLP 응용 프로그램에서 중요한 단계입니다. 예를 들어, 스팸 메일 분류, 문서 군집화, 텍스트 기반 추천 시스템 등에 사용됩니다.
#
# - CountVectorizer 주요 파라미터  
#     - min_df : vocabulary 에 포함할 최소 발생 빈도. 어떤 단어가 너무 드물게 나타나면 무시하고 싶을 때 사용
#     - ngram_range : 단어를 몇 개씩 묶어서 볼 것인지 정합니다. (1, 1) - unigram only, (1, 2) - unigram + bigram
#     - max_features : 자주 등장하는 단어 중 상위 몇 개까지만 사용할지 정합니다.  
#     - token_pattern = (?u)\\b\\w\\w+\\b : 단어로 인정할 기준. unocode 영수자 2 글자 이상만 포함

# %% [markdown] id="00iVGIN7bmv1"
# ## Text vs token Matrix 생성

# %% colab={"base_uri": "https://localhost:8080/"} id="Yh1c2pDZbmv1" outputId="edca934c-2cb2-48db-c531-10768eb6eced"
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer 객체 생성
count_vectorizer = CountVectorizer()

# sentences 데이터에 대한 피처 변환 수행
# sentences는 분석할 텍스트 데이터의 리스트
features = count_vectorizer.fit_transform(sentences)
features

# %% colab={"base_uri": "https://localhost:8080/"} id="9Fqsw6scbmv1" outputId="a3952adf-7009-45e4-9300-4d1ef4cd7243"
print(f"document 수: {features.shape[0]}")
print(f"단어수: {features.shape[1]}")

# %% colab={"base_uri": "https://localhost:8080/"} id="VSpUvWqcbmv2" outputId="18a86fee-b01f-4f9d-bcb7-725496a70e37"
# features 객체를 NumPy 배열로 변환
vectorized_sentences = features.toarray()
vectorized_sentences

# %% [markdown] id="kLr_wKbibmv2"
# ### features 의 단어 list

# %% colab={"base_uri": "https://localhost:8080/"} id="KJpQEuXlbmv2" outputId="59602c7b-4ef4-47c1-b275-0578d2e3619c"
# CountVectorizer를 통해 추출한 피처(단어) 이름들을 가져옴
feature_names = count_vectorizer.get_feature_names_out()
feature_names

# %% colab={"base_uri": "https://localhost:8080/", "height": 237} id="xVukxEFKbmv2" outputId="06bdd911-2bdd-49ff-9c15-923cb1bb7560"
# 벡터화된 문장과 피처 이름을 이용해 DataFrame 생성
df = pd.DataFrame(vectorized_sentences, columns=feature_names)

# 데이터프레임의 인덱스 이름 지정
df.index.name = 'sentence'
df

# %% [markdown] id="WSCEeO7Abmv2"
# ## 2. TF-IDF
#
# - TF-IDF(Term Frequency - Inverse Document Frequency)  
#
# TF-IDF는 단어의 빈도와 그 단어가 드물게 나타나는 문서에 더 높은 가중치를 부여하는 방식으로 작동합니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="dFRJhDZXbmv2" outputId="b9a05124-324d-4c1f-a007-6febd9c6e420"
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# sentences 데이터에 대한 TF-IDF 기반 피처 변환 수행
tfidf_sentences = tfidf_vectorizer.fit_transform(sentences)
tfidf_sentences

# %% [markdown] id="7_YNzGsIbmv3"
# ## Text vs tf-idf Matrix 생성

# %% colab={"base_uri": "https://localhost:8080/"} id="MuEA8J4Sbmv3" outputId="054ccbac-6611-460b-a3b2-e1f23b0b51b0"
# TF-IDF 피처 객체를 NumPy 배열로 변환
tfidf_vect_sentences = tfidf_sentences.toarray()
tfidf_vect_sentences

# %% colab={"base_uri": "https://localhost:8080/"} id="ONf1SaY3bmv3" outputId="6ede79c4-3582-46f0-b58f-a31cc660d614"
# TfidfVectorizer를 통해 추출한 피처(단어) 이름들을 가져옴
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_feature_names

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="XHSBD5f0bmv3" outputId="4ede32ec-9317-457c-c06d-ff7ccd4a7c39"
# TF-IDF 벡터화된 문장과 피처 이름을 이용해 DataFrame 생성
df = pd.DataFrame(tfidf_vect_sentences, columns=tfidf_feature_names)
df

# %% [markdown] id="GJIYRQeEbmv3"
# # 3. keras word encoding
#
# - keras  API 이용

# %% id="Vjek94uvbmv3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# %% [markdown] id="F_3RVyplbmv3"
# ## Tokenize

# %% id="GftMA-eobmv3"
# 문장으로 부터 상위 100 개 단어로 vocabulary 작성
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# %% [markdown] id="Mkinjq-Xbmv3"
# ## Word Index Vocabulary 작성

# %% colab={"base_uri": "https://localhost:8080/"} id="7oGvaVHPbmv4" outputId="1b4bca65-9bc5-4ede-fcd3-1673a8f3dfc4"
# sentences에 포함된 문장들을 기반으로 단어의 토큰화를
# 수행하며, 각 단어에 고유한 인덱스를 할당
tokenizer.fit_on_texts(sentences)

# 각 단어에 부여된 고유 인덱스 추출
print(tokenizer.word_index)
print(tokenizer.index_word)

# %% [markdown] id="lV07Led4bmv4"
# ## text 의 sentence 변환 및 paddding
#
# - texts_to_sequences: text list 내의 각 text 를 수열 (sequence of integers) 로 convert
#
#
#     - 입력 : text (strings) list
#     - 반환 : sequence list
#     
# - pad_sequences: 동일한 길이로 sequence 를 zero padding

# %% colab={"base_uri": "https://localhost:8080/"} id="o7fAurihbmv4" outputId="303cec05-d213-4dcf-f3b1-8b51989e8411"
# sentences 데이터를 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(sentences)

# 시퀀스에 패딩 적용 (문장의 뒤쪽을 패딩하고, 필요시 뒤쪽을 잘라냄)
padded = pad_sequences(sequences, padding='post', truncating='post')

print(sequences)
print()
print(padded)

# %% [markdown] id="sMjZr__Wbmv4"
# ### sequenced sentence 를 word sentence 로 환원

# %% colab={"base_uri": "https://localhost:8080/"} id="RRPb29Whbmv4" outputId="6d652de0-6ac5-4318-b685-820e6c134ac2"
# sequences 리스트에 있는 각 시퀀스를 처리
for sequence in sequences:
    sent = []          # 문장을 저장할 리스트 초기화
    for idx in sequence:
        sent.append(tokenizer.index_word[idx])   # 인덱스를 단어로 변환하여 문장에 추가
    print(' '.join(sent))      # 단어 리스트를 공백으로 연결하여 문장으로 만들고 출력


# %% [markdown] id="2Gs4XtP4bmv4"
# ### One-Hot-Encoding 표현

# %% colab={"base_uri": "https://localhost:8080/"} id="N_9vzATxbmv4" outputId="45d83dc7-a376-4d94-b6f5-9249a3648f24"
# 패딩된 시퀀스를 원-핫 인코딩으로 변환
to_categorical(padded)

# %% id="Xfo5BECvbmv4"
