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
# # 프로젝트 B: 딥러닝 기반 영화 리뷰 감성 분석 시스템
#
# ## 프로젝트 목표
# - 딥러닝 기법을 활용한 감성 분석 시스템 구축
# - 두 가지 방법 비교: Hugging Face Pipeline vs Gemini API
# - 네이버 영화평 데이터를 이용한 실전 감성 분석
#
# ## 학습 내용
# 1. 최소한의 데이터 전처리 (딥러닝 모델용)
# 2. 감성 분석 방법 1: Hugging Face Pipeline (BERT)
# 3. 감성 분석 방법 2: Gemini API
# 4. 두 방법 비교 및 평가

# %% [markdown]
# ---
# ## 1. 데이터 준비 및 최소 전처리

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import pipeline
from google import genai
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 환경 변수 로드
load_dotenv()

# Gemini API 클라이언트 초기화
try:
    client = genai.Client()
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Gemini API 초기화 실패: {e}")
    print("   .env 파일에 GOOGLE_API_KEY가 설정되어 있는지 확인하세요.")
    GEMINI_AVAILABLE = False

# 네이버 영화평 데이터 다운로드
DATA_TRAIN_PATH = tf.keras.utils.get_file(
    "ratings_train.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt"
)
DATA_TEST_PATH = tf.keras.utils.get_file(
    "ratings_test.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt"
)

# 데이터 로드
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter='\t')
test_data = pd.read_csv(DATA_TEST_PATH, delimiter='\t')

print("=" * 80)
print("[데이터 로드 완료]")
print("=" * 80)
print(f"훈련 데이터: {train_data.shape}")
print(f"테스트 데이터: {test_data.shape}")

# 결측값 제거
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# 데이터 샘플링
df_train = train_data.sample(n=5_000, random_state=1)
df_test = test_data.sample(n=1_000, random_state=1)

print(f"\n샘플링 후:")
print(f"훈련 데이터: {df_train.shape}")
print(f"테스트 데이터: {df_test.shape}")

# 딥러닝 모델용 최소 전처리 (결측값 제거만)
df_train['cleaned_document'] = df_train['document'].astype(str)
df_test['cleaned_document'] = df_test['document'].astype(str)

# 빈 문자열 제거
df_train = df_train[df_train['cleaned_document'].str.len() > 0]
df_test = df_test[df_test['cleaned_document'].str.len() > 0]

print(f"\n전처리 후:")
print(f"훈련 데이터: {df_train.shape}")
print(f"테스트 데이터: {df_test.shape}")

# %% [markdown]
# ---
# ## 2. 감성 분석 방법 1: Hugging Face Pipeline

# %% [markdown]
# ### 2.1 파이프라인을 이용한 감성 분석

# %%
print("=" * 80)
print("[Hugging Face 감성 분석]")
print("=" * 80)

# 다국어 감성 분석 모델
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_classifier = pipeline('sentiment-analysis', model=model_name)

# 샘플 리뷰 분석
sample_texts = [
    "다시는 보고 싶지 않은 짜증나는 영화",
    "아주 재미있는 영화",
    "정말 재미없는 영화였다",
    "이 영화 최고",
    "보통 영화"
]

print("\n[감성 분석 결과]")
results = sentiment_classifier(sample_texts)

for i, result in enumerate(results):
    print(f"{sample_texts[i]}")
    print(f"  → {result['label']}, 신뢰도: {result['score']:.4f}\n")

# %% [markdown]
# ### 2.2 테스트 데이터 감성 분석

# %%
print("=" * 80)
print("[테스트 데이터 감성 분석]")
print("=" * 80)

# 테스트 데이터 샘플 분석
test_samples = df_test['cleaned_document'].head(10).tolist()
test_labels = df_test['label'].head(10).tolist()

print("\n[테스트 데이터 샘플 분석]")
sentiment_results = sentiment_classifier(test_samples)

correct = 0
for i, (text, true_label, result) in enumerate(zip(test_samples, test_labels, sentiment_results)):
    # 별점을 긍정/부정으로 변환 (4-5점: 긍정, 1-2점: 부정)
    predicted_label = 1 if int(result['label'].split()[0]) >= 4 else 0
    true_label_str = "긍정" if true_label == 1 else "부정"
    predicted_label_str = "긍정" if predicted_label == 1 else "부정"
    
    is_correct = "✓" if true_label == predicted_label else "✗"
    if true_label == predicted_label:
        correct += 1
    
    print(f"\n리뷰 {i+1}: {text[:50]}...")
    print(f"  실제: {true_label_str}, 예측: {predicted_label_str} ({result['label']}) {is_correct}")

accuracy_hf = correct / len(test_samples)
print(f"\n정확도: {accuracy_hf:.4f} ({correct}/{len(test_samples)})")

# %% [markdown]
# ---
# ## 3. 감성 분석 방법 2: Gemini API

# %% [markdown]
# ### 3.1 Gemini를 사용한 감성 분석
#
# **지시사항**:
# - Gemini API를 사용하여 감성 분석 수행
# - 프롬프트를 통해 긍정/부정 분류 요청
# - Hugging Face와 동일한 테스트 데이터로 평가
#
# %%
if GEMINI_AVAILABLE:
    print("=" * 80)
    print("[Gemini API 감성 분석]")
    print("=" * 80)
    
    def analyze_sentiment_gemini(text):
        """
        Gemini API를 사용한 감성 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감성 분석 결과 (긍정/부정)
        """
        prompt = f"""다음 영화 리뷰의 감성을 분석하여 긍정 또는 부정으로 분류해주세요.
        
리뷰: {text}

답변 형식: 긍정 또는 부정만 출력하세요."""
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            result_text = response.text.strip()
            # "긍정" 또는 "부정" 키워드 확인
            if "긍정" in result_text:
                return 1
            elif "부정" in result_text:
                return 0
            else:
                # 기본값 (긍정으로 가정)
                return 1
        except Exception as e:
            print(f"오류 발생: {e}")
            return None
    
    # 샘플 리뷰 분석
    sample_texts = [
        "다시는 보고 싶지 않은 짜증나는 영화",
        "아주 재미있는 영화",
        "정말 재미없는 영화였다",
        "이 영화 최고",
        "보통 영화"
    ]
    
    print("\n[감성 분석 결과]")
    for text in sample_texts:
        result = analyze_sentiment_gemini(text)
        sentiment = "긍정" if result == 1 else "부정"
        print(f"{text} → {sentiment}")

# %% [markdown]
#     # ### 3.2 테스트 데이터 감성 분석
#     #
    # %%
    print("=" * 80)
    print("[테스트 데이터 감성 분석 (Gemini)]")
    print("=" * 80)
    
    # 테스트 데이터 샘플 분석
    test_samples = df_test['cleaned_document'].head(10).tolist()
    test_labels = df_test['label'].head(10).tolist()
    
    print("\n[테스트 데이터 샘플 분석]")
    gemini_results = []
    
    correct_gemini = 0
    for i, (text, true_label) in enumerate(zip(test_samples, test_labels)):
        predicted_label = analyze_sentiment_gemini(text)
        
        if predicted_label is not None:
            true_label_str = "긍정" if true_label == 1 else "부정"
            predicted_label_str = "긍정" if predicted_label == 1 else "부정"
            
            is_correct = "✓" if true_label == predicted_label else "✗"
            if true_label == predicted_label:
                correct_gemini += 1
            
            print(f"\n리뷰 {i+1}: {text[:50]}...")
            print(f"  실제: {true_label_str}, 예측: {predicted_label_str} {is_correct}")
            
            gemini_results.append(predicted_label)
        else:
            gemini_results.append(None)
    
    accuracy_gemini = correct_gemini / len(test_samples)
    print(f"\n정확도: {accuracy_gemini:.4f} ({correct_gemini}/{len(test_samples)})")

else:
    print("=" * 80)
    print("[Gemini API 감성 분석]")
    print("=" * 80)
    print("⚠️ Gemini API가 설정되지 않아 Gemini 감성 분석을 건너뜁니다.")
    print("   .env 파일에 GOOGLE_API_KEY를 설정하세요.")
    accuracy_gemini = None

# %% [markdown]
# ---
# ## 4. 두 방법 비교 및 평가

# %%
print("=" * 80)
print("[두 방법 비교]")
print("=" * 80)

comparison_data = {
    '구분': [
        '방법',
        '모델',
        '정확도',
        '처리 속도',
        '설치 필요',
        'API 키 필요',
        '장점',
        '단점'
    ],
    'Hugging Face Pipeline': [
        '사전 학습된 BERT 모델',
        'nlptown/bert-base-multilingual-uncased-sentiment',
        f'{accuracy_hf:.4f}' if 'accuracy_hf' in locals() else 'N/A',
        '빠름 (로컬 실행)',
        '✅ transformers 라이브러리',
        '❌ 불필요',
        '빠른 처리, 오프라인 사용 가능',
        '모델 다운로드 필요, GPU 권장'
    ],
    'Gemini API': [
        'LLM 기반',
        'gemini-2.5-flash',
        f'{accuracy_gemini:.4f}' if accuracy_gemini is not None else 'N/A',
        '상대적으로 느림 (API 호출)',
        '❌ 불필요',
        '✅ GOOGLE_API_KEY 필요',
        '프롬프트 조정 가능, 유연함',
        '인터넷 연결 필요, API 비용'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n")
print(df_comparison.to_string(index=False))

# %% [markdown]
# ---
# ## 5. 프로젝트 정리

# %%
print("=" * 80)
print("[프로젝트 정리]")
print("=" * 80)

summary = """
프로젝트 B: 딥러닝 기반 영화 리뷰 감성 분석 시스템

1. 최소한의 데이터 전처리
   - 결측값 제거만 수행
   - 딥러닝 모델이 원본 패턴을 학습했으므로 과도한 전처리 불필요

2. 감성 분석 방법 1: Hugging Face Pipeline
   - 사전 학습된 BERT 모델 활용
   - 파이프라인을 통한 간편한 사용
   - 빠른 처리 속도
   - 오프라인 사용 가능

3. 감성 분석 방법 2: Gemini API
   - LLM 기반 감성 분석
   - 프롬프트 조정으로 유연한 분석
   - API를 통한 간편한 사용

4. 두 방법 비교
   - 정확도 비교
   - 처리 속도 비교
   - 사용 편의성 비교

특징:
- 두 가지 방법 비교 학습
- 실전 감성 분석 경험
- 각 방법의 장단점 이해
"""

print(summary)

# %% [markdown]
# ---
# ### 다음 단계
# - 프로젝트 A (TF-IDF)와 결과 비교
# - 프로젝트 C (RAG)로 질의응답 시스템 구축
# - 실무 적용 시나리오 고려

# %%

