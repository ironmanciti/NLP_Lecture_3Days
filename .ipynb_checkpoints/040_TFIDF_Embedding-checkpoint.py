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

# %% [markdown]
# # 문장 벡터화: BOW, TF-IDF vs 임베딩
#
# ## 학습 목표
# - BOW (Bag of Words) 벡터화 방법 이해
# - TF-IDF 벡터화 방법 이해
# - TF-IDF 벡터 기반 유사도 계산 (cosine_similarity)
# - Sentence Transformer 임베딩 방법 이해
# - 임베딩 벡터 기반 유사도 계산
# - 두 방식의 차이점 이해 (단어 매칭 vs 의미 유사도)
#
# ## 학습 내용
# 1. BOW (Bag of Words) 벡터화
# 2. TF-IDF 벡터화 및 유사도 계산
# 3. Sentence Transformer 임베딩 및 유사도 계산
# 4. 방법론 비교: 단어 매칭 vs 의미 유사도

# %% [markdown]
# ---
# ## 1. 데이터 준비

# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 금융 뉴스 예시 데이터
finance_news = [
    "삼성전자 3분기 영업이익 급등, 사상최고 실적 기대",
    "코스피 하락세 지속, 외국인 순매도 확대에 우려",
    "반도체 시장 성장세 지속, 메모리 반도체 수요 증가",
    "삼성전자 주가 상승, 실적 호조 전망 낙관",
    "금리 인상 우려로 주식 시장 하락"
]

print("=" * 80)
print("금융 뉴스 데이터")
print("=" * 80)
for i, news in enumerate(finance_news, 1):
    print(f"{i}. {news}")

# %% [markdown]
# ---
# ## 2. BOW (Bag of Words) 벡터화
#
# **BOW의 특징:**
# - 단어의 순서를 무시하고 단어 빈도만 고려
# - 희소 벡터(Sparse Vector) 생성
# - 빠른 처리 속도
# - 의미적 유사도 파악 어려움

# %%
print("\n" + "=" * 80)
print("[BOW - Bag of Words 벡터화]")
print("=" * 80)

# CountVectorizer 객체 생성
count_vectorizer = CountVectorizer()

# 문장들을 벡터로 변환
bow_features = count_vectorizer.fit_transform(finance_news)
bow_array = bow_features.toarray()

# 단어 목록
feature_names = count_vectorizer.get_feature_names_out()

print(f"\n문서 수: {bow_features.shape[0]}")
print(f"단어 수: {bow_features.shape[1]}")
print(f"\n단어 목록 (일부): {list(feature_names[:15])}...")

# DataFrame으로 시각화
df_bow = pd.DataFrame(
    bow_array, 
    columns=feature_names, 
    index=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[BOW 벡터 행렬]")
print(df_bow)

# %% [markdown]
# ---
# ## 3. TF-IDF 벡터화 및 유사도 계산
#
# **TF-IDF의 특징:**
# - 단어의 중요도를 문서 내 빈도와 전체 문서에서의 희귀도를 고려
# - BOW보다 의미 있는 가중치 부여
# - 여전히 희소 벡터이지만 BOW보다 정보량이 많음

# %%
print("\n" + "=" * 80)
print("[TF-IDF 벡터화]")
print("=" * 80)

# TfidfVectorizer 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# 문장들을 TF-IDF 벡터로 변환
tfidf_features = tfidf_vectorizer.fit_transform(finance_news)
tfidf_array = tfidf_features.toarray()

# 단어 목록
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\n문서 수: {tfidf_features.shape[0]}")
print(f"단어 수: {tfidf_features.shape[1]}")

# DataFrame으로 시각화
df_tfidf = pd.DataFrame(
    tfidf_array, 
    columns=tfidf_feature_names, 
    index=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[TF-IDF 벡터 행렬]")
print(df_tfidf.round(3))

# %% [markdown]
# ### 3.1 TF-IDF 벡터 기반 유사도 계산 (cosine_similarity)

# %%
print("\n" + "=" * 80)
print("[TF-IDF 벡터 기반 유사도 계산 - cosine_similarity]")
print("=" * 80)

# TF-IDF 벡터 간 코사인 유사도 계산
tfidf_similarity = cosine_similarity(tfidf_features)

# 유사도 행렬을 DataFrame으로 변환
df_tfidf_sim = pd.DataFrame(
    tfidf_similarity,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[TF-IDF 기반 코사인 유사도 행렬]")
print(df_tfidf_sim.round(3))

# 뉴스 쌍별 유사도 출력
print("\n[뉴스 쌍별 유사도]")
for i in range(len(finance_news)):
    for j in range(i+1, len(finance_news)):
        sim = tfidf_similarity[i][j]
        print(f"  뉴스{i+1} vs 뉴스{j+1}: {sim:.4f}")
        print(f"    '{finance_news[i]}'")
        print(f"    '{finance_news[j]}'")
        print()

# %% [markdown]
# ---
# ## 4. Sentence Transformer 임베딩 및 유사도 계산
#
# **임베딩의 특징:**
# - 문장 전체를 고정 크기의 밀집 벡터(Dense Vector)로 변환
# - 문맥과 의미를 이해하여 유사한 의미의 문장은 유사한 벡터 생성
# - 단어 순서와 문맥을 고려
# - 계산 비용이 높지만 정확도가 높음

# %%
print("\n" + "=" * 80)
print("[Sentence Transformer 임베딩]")
print("=" * 80)

# KURE-v1 모델 로드 (한국어 특화)
print("\n임베딩 모델 로드 중...")
model = SentenceTransformer("nlpai-lab/KURE-v1")
print("✓ KURE-v1 모델 로드 완료 (한국어 특화 문장 임베딩 모델)")

# 문장들을 벡터로 변환
embeddings = model.encode(finance_news)

print(f"\n임베딩 차원: {embeddings.shape}")
print(f"  - 문장 수: {embeddings.shape[0]}")
print(f"  - 벡터 차원: {embeddings.shape[1]}")

# 첫 번째 문장의 임베딩 벡터 일부 확인
print(f"\n첫 번째 뉴스 임베딩 벡터 (처음 10개 값):")
print(embeddings[0][:10])

# %% [markdown]
# ### 4.1 임베딩 벡터 기반 유사도 계산

# %%
print("\n" + "=" * 80)
print("[임베딩 벡터 기반 유사도 계산]")
print("=" * 80)

# 방법 1: model.similarity() 사용
embedding_similarity_model = model.similarity(embeddings, embeddings)

# 방법 2: cosine_similarity() 직접 사용 (동일한 결과)
embedding_similarity_cosine = cosine_similarity(embeddings)

print("\n[방법 1: model.similarity() 사용]")
df_embed_sim_model = pd.DataFrame(
    embedding_similarity_model.numpy(),
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print(df_embed_sim_model.round(3))

print("\n[방법 2: cosine_similarity() 직접 사용]")
df_embed_sim_cosine = pd.DataFrame(
    embedding_similarity_cosine,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print(df_embed_sim_cosine.round(3))

print("\n💡 두 방법 모두 동일한 코사인 유사도를 계산합니다!")

# 뉴스 쌍별 유사도 출력
print("\n[뉴스 쌍별 유사도]")
for i in range(len(finance_news)):
    for j in range(i+1, len(finance_news)):
        sim = embedding_similarity_cosine[i][j]
        print(f"  뉴스{i+1} vs 뉴스{j+1}: {sim:.4f}")
        print(f"    '{finance_news[i]}'")
        print(f"    '{finance_news[j]}'")
        print()

# %% [markdown]
# ---
# ## 5. 방법론 비교: 단어 매칭 vs 의미 유사도
#
# ### 5.1 TF-IDF와 임베딩의 차이점 비교

# %%
print("=" * 80)
print("[TF-IDF vs 임베딩: 단어 매칭 vs 의미 유사도 비교]")
print("=" * 80)

# 테스트 케이스: 의미는 같지만 단어가 다른 문장들
test_cases = [
    {
        'name': '의미 동일, 단어 다름',
        'text1': '삼성전자 주가가 상승했습니다',
        'text2': '삼성전자 주식 가격이 올랐습니다'  # 의미는 같지만 단어가 다름
    },
    {
        'name': '의미 유사, 표현 다름',
        'text1': '주가 상승 실적 호조 전망 낙관',
        'text2': '주식 가격 증가 실적 좋음 전망 긍정적'  # 의미 유사하지만 표현 다름
    },
    {
        'name': '의미 다름',
        'text1': '삼성전자 주가 상승',
        'text2': '코스피 지수 하락'  # 완전히 다른 의미
    },
    {
        'name': '단어 일부 겹침',
        'text1': '반도체 시장 성장',
        'text2': '반도체 수요 증가'  # 일부 단어 겹침
    }
]

# 결과 저장
results = []

for case in test_cases:
    text1 = case['text1']
    text2 = case['text2']
    
    # === TF-IDF 방식 ===
    tfidf_test = TfidfVectorizer()
    tfidf_vectors = tfidf_test.fit_transform([text1, text2])
    tfidf_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]
    
    # 공통 단어 확인
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1.intersection(words2)
    
    # === 임베딩 방식 ===
    emb_vectors = model.encode([text1, text2])
    emb_sim = cosine_similarity([emb_vectors[0]], [emb_vectors[1]])[0][0]
    
    results.append({
        '케이스': case['name'],
        '문장1': text1,
        '문장2': text2,
        '공통단어': ', '.join(common_words) if common_words else '없음',
        'TF-IDF_유사도': tfidf_sim,
        '임베딩_유사도': emb_sim,
        '차이': emb_sim - tfidf_sim
    })
    
    # 상세 출력
    print("-" * 80)
    print(f"[케이스: {case['name']}]")
    print(f"  문장1: {text1}")
    print(f"  문장2: {text2}")
    print(f"  공통 단어: {', '.join(common_words) if common_words else '없음'}")
    print(f"\n  TF-IDF 유사도: {tfidf_sim:.4f} {'(단어 매칭 중심)' if tfidf_sim < 0.3 else '(단어 겹침)'}")
    print(f"  임베딩 유사도: {emb_sim:.4f} {'(의미 유사도 중심)' if emb_sim > 0.5 else '(의미 다름)'}")
    print(f"  차이: {emb_sim - tfidf_sim:+.4f}")
    
    # 해석
    if tfidf_sim < 0.3 and emb_sim > 0.5:
        print(f"\n  💡 해석: 단어가 다르지만 의미가 유사 → TF-IDF는 낮게, 임베딩은 높게 평가")
    elif tfidf_sim > 0.3 and emb_sim < 0.5:
        print(f"\n  💡 해석: 단어는 겹치지만 의미가 다름 → TF-IDF는 높게, 임베딩은 낮게 평가")
    print()

# 결과 요약 테이블
print("=" * 80)
print("[결과 요약]")
print("=" * 80)

df_results = pd.DataFrame(results)
print("\n[상세 결과]")
print(df_results[['케이스', 'TF-IDF_유사도', '임베딩_유사도', '차이']].to_string(index=False))

# %% [markdown]
# ### 5.2 핵심 차이점 요약

# %%
print("\n" + "=" * 80)
print("[핵심 차이점 요약]")
print("=" * 80)

comparison_data = {
    '특성': [
        '벡터 타입',
        '벡터 차원',
        '유사도 기준',
        '단어 순서 고려',
        '계산 속도',
        '의미 이해',
        '주요 활용 분야'
    ],
    'TF-IDF': [
        '희소 벡터 (Sparse)',
        '어휘 크기에 따라 가변',
        '단어 매칭 중심',
        '❌ 무시',
        '⚡ 매우 빠름',
        '제한적 (빈도 기반)',
        '키워드 추출, 문서 분류'
    ],
    '임베딩': [
        '밀집 벡터 (Dense)',
        '고정 (1024차원)',
        '의미 유사도 중심',
        '✅ 고려',
        '🐌 느림',
        '우수 (문맥 기반)',
        '유사도 검색, 감성 분석'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n")
print(df_comparison.to_string(index=False))

print("""
\n[핵심 정리]

1. TF-IDF (단어 매칭 중심)
   ✓ 단어가 겹치면 유사도 높음
   ✗ 단어가 다르면 유사도 낮음 (의미가 같아도)
   → "주가 상승" vs "주식 가격 증가" → 낮은 유사도

2. 임베딩 (의미 유사도 중심)
   ✓ 의미가 같으면 유사도 높음 (단어가 달라도)
   ✗ 의미가 다르면 유사도 낮음 (단어가 겹쳐도)
   → "주가 상승" vs "주식 가격 증가" → 높은 유사도
""")

# %% [markdown]
# ---
# ## 6. 학습 정리
#
# ### 6.1 벡터화 방법 요약
#
# | 방법 | 벡터화 함수 | 유사도 계산 함수 | 특징 |
# |------|------------|----------------|------|
# | BOW | `CountVectorizer().fit_transform()` | `cosine_similarity()` | 단어 빈도 기반 |
# | TF-IDF | `TfidfVectorizer().fit_transform()` | `cosine_similarity()` | 단어 중요도 기반 |
# | 임베딩 | `SentenceTransformer().encode()` | `cosine_similarity()` 또는 `model.similarity()` | 의미 기반 |
#
# ### 6.2 언제 어떤 방법을 사용할까?
#
# | 상황 | 추천 방법 | 이유 |
# |------|----------|------|
# | 빠른 키워드 추출 | BOW/TF-IDF | 빠른 처리, 해석 용이 |
# | 문서 분류 (단어 매칭 중요) | TF-IDF | 전통적 ML 모델과 호환 |
# | 의미 기반 검색 | 임베딩 | 문맥 이해, 유사 의미 인식 |
# | 감성 분석 | 임베딩 | 문맥 기반 감성 파악 |
# | 대량 문서 처리 (속도 중요) | TF-IDF | 빠른 처리 속도 |
# | 정확한 의미 분석 (정확도 중요) | 임베딩 | 높은 정확도 |

# %%
print("=" * 80)
print("[학습 정리]")
print("=" * 80)

summary = """
1. BOW (Bag of Words)
   - CountVectorizer 사용
   - 단어 빈도 기반 희소 벡터
   - 빠르지만 의미 이해 제한적

2. TF-IDF
   - TfidfVectorizer 사용
   - 단어 중요도 기반 희소 벡터
   - cosine_similarity()로 유사도 계산
   - 단어 매칭 중심 (단어가 다르면 유사도 낮음)

3. 임베딩 (Sentence Transformer)
   - SentenceTransformer.encode() 사용
   - 의미 기반 밀집 벡터
   - cosine_similarity() 또는 model.similarity()로 유사도 계산
   - 의미 유사도 중심 (의미가 같으면 유사도 높음)

4. 실무 적용
   - 상황에 따라 적절한 방법 선택
   - TF-IDF: 빠른 처리, 키워드 추출
   - 임베딩: 정확한 의미 분석, 감성 분석
"""

print(summary)

# %% [markdown]
# ---
# ### 다음 단계
# - 25차시: 자연어 처리 기초 심화
# - 26차시: 금융 뉴스 감성 분석 실습
# - 300차시: Hugging Face 파이프라인 활용

# %%

