# 프로젝트 A: TF-IDF 기반 영화 리뷰 감성 분석 시스템 (실습 문제)

## 📋 실습 목표
- 전통적 NLP 기법(TF-IDF + sklearn)을 활용한 감성 분석 시스템을 직접 구현
- 네이버 영화평 데이터를 이용한 실전 감성 분석 수행
- Logistic Regression 분류기 학습 및 평가

## 📚 참고 자료
- **정답 코드**: `070_Project_Movie_Review_TFIDF.py` (참고용)
- **전처리 방법**: `050_Data_Preprocessing.py` 섹션 2 참고
- **TF-IDF 벡터화**: `040_TFIDF_Embedding.py` 섹션 3 참고

---

## 단계별 실습 문제

### 🔹 단계 1: 데이터 준비 및 로드

**목표**: 네이버 영화평 데이터를 다운로드하고 로드하기

**지시사항**:
1. 필요한 라이브러리를 import하세요:
   - `pandas`, `numpy`, `re`, `tensorflow`
   - `sklearn`에서 `TfidfVectorizer`, `LogisticRegression` import
   - `sklearn.metrics`에서 `accuracy_score`, `confusion_matrix`, `classification_report` import

2. `tensorflow.keras.utils.get_file()`을 사용하여 다음 URL에서 데이터를 다운로드하세요:
   - 훈련 데이터: `https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt`
   - 테스트 데이터: `https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt`

3. `pd.read_csv()`를 사용하여 데이터를 로드하세요 (구분자는 `'\t'`)

4. 결측값을 제거하세요 (`dropna()` 사용)

5. 처리 속도 향상을 위해 데이터를 샘플링하세요:
   - 훈련 데이터: 10,000개 (random_state=1)
   - 테스트 데이터: 3,000개 (random_state=1)

6. 레이블 분포를 확인하세요 (`value_counts()` 사용)

**참고**: `070_Project_Movie_Review_TFIDF.py` 섹션 1 참고

---

### 🔹 단계 2: 데이터 전처리

**목표**: 텍스트 정제 및 불용어 제거 함수 작성 및 적용

**지시사항**:

#### 2.1 텍스트 정제 함수 작성
1. `clean_text_basic(text)` 함수를 작성하세요. 다음 기능을 포함해야 합니다:
   - HTML 태그 제거 (정규식: `r'<[^>]+>'`)
   - URL 제거 (정규식: `r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'`)
   - 이메일 제거 (정규식: `r'\S+@\S+'`)
   - 전화번호 제거 (정규식: `r'\d{2,3}-\d{3,4}-\d{4}'`)
   - 한글, 영문, 숫자만 유지하고 나머지 특수문자 제거 (정규식: `r'[^가-힣a-zA-Z0-9\s]'`)
   - 중복 공백 정리 (정규식: `r'\s+'` → `' '`로 치환)
   - `strip()`으로 앞뒤 공백 제거

2. 함수 내에서 `pd.isna(text)`로 결측값을 체크하고, 결측값이면 빈 문자열을 반환하세요.

**참고**: `050_Data_Preprocessing.py` 섹션 2.1의 `clean_text_basic` 함수 참고

#### 2.2 불용어 제거 함수 작성
1. 한국어 불용어 리스트를 정의하세요 (최소 20개 이상):
   ```python
   korean_stopwords = ['이', '가', '을', '를', ...]
   ```

2. `remove_stopwords(text, stopwords_list)` 함수를 작성하세요:
   - 텍스트를 공백으로 split하여 단어 리스트 생성
   - 불용어 리스트에 없는 단어만 필터링
   - 필터링된 단어들을 다시 공백으로 join하여 반환

**참고**: `050_Data_Preprocessing.py` 섹션 3.1의 불용어 제거 함수 참고

#### 2.3 전처리 적용
1. `df_train`과 `df_test`에 `clean_text_basic()` 함수를 적용하여 `cleaned_document` 컬럼을 생성하세요 (`apply()` 사용)

2. `remove_stopwords()` 함수를 적용하여 불용어를 제거하세요

3. 빈 문자열을 가진 행을 제거하세요 (`str.len() > 0` 조건 사용)

4. 전처리 결과를 확인하기 위해 샘플 3개를 출력하세요 (원본과 정제된 텍스트 비교)

**참고**: `070_Project_Movie_Review_TFIDF.py` 섹션 2.3 참고

---

### 🔹 단계 3: TF-IDF 벡터화

**목표**: 텍스트를 TF-IDF 벡터로 변환하기

**지시사항**:
1. `TfidfVectorizer` 객체를 생성하세요. 다음 파라미터를 사용하세요:
   - `max_features=5000`: 최대 특성 수
   - `ngram_range=(1, 2)`: 1-gram과 2-gram 사용
   - `min_df=2`: 최소 문서 빈도
   - `max_df=0.95`: 최대 문서 빈도

2. 훈련 데이터로 `fit_transform()`을 사용하여 벡터화하세요

3. 테스트 데이터로 `transform()`을 사용하여 벡터화하세요 (주의: `fit_transform()`이 아님!)

4. 레이블을 추출하세요:
   - `y_train = df_train['label'].values`
   - `y_test = df_test['label'].values`

5. 벡터의 shape와 특성 수를 출력하세요

**참고**: 
- `040_TFIDF_Embedding.py` 섹션 3의 `TfidfVectorizer` 사용법 참고
- `070_Project_Movie_Review_TFIDF.py` 섹션 3 참고

---

### 🔹 단계 4: Logistic Regression 분류기 학습 및 평가

**목표**: Logistic Regression 모델을 학습하고 성능을 평가하기

**지시사항**:
1. `LogisticRegression` 객체를 생성하세요:
   - `random_state=42` (재현성을 위해)
   - `max_iter=1000` (수렴을 위해)

2. `fit()` 메서드를 사용하여 모델을 학습하세요:
   - 입력: `X_train_tfidf` (TF-IDF 벡터)
   - 출력: `y_train` (레이블)

3. `predict()` 메서드를 사용하여 테스트 데이터에 대한 예측을 수행하세요

4. 성능을 평가하세요:
   - `accuracy_score()`로 정확도 계산
   - `confusion_matrix()`로 혼동 행렬 출력
   - `classification_report()`로 분류 리포트 출력 (target_names=['부정', '긍정'])

**참고**: `070_Project_Movie_Review_TFIDF.py` 섹션 4 참고

---

### 🔹 단계 5: (선택) 키워드 분석

**목표**: 긍정/부정 키워드 추출 및 시각화

**지시사항**:
1. `tfidf_vectorizer.get_feature_names_out()`로 특성 이름(단어)을 가져오세요

2. `lr_model.coef_[0]`로 Logistic Regression의 계수를 가져오세요

3. 계수가 큰 순서대로 정렬하여 긍정 키워드 Top 20을 추출하세요

4. 계수가 작은 순서대로 정렬하여 부정 키워드 Top 20을 추출하세요

5. (선택) `matplotlib`을 사용하여 키워드를 시각화하세요

**참고**: `070_Project_Movie_Review_TFIDF.py` 섹션 5 참고 (정답 코드에 포함되어 있지 않지만, 추가 학습용)

---

## ✅ 체크리스트

각 단계를 완료한 후 다음을 확인하세요:

- [ ] 단계 1: 데이터가 정상적으로 로드되고 샘플링되었는가?
- [ ] 단계 2: 전처리 함수가 올바르게 작동하는가? (HTML, URL, 이메일 등이 제거되는가?)
- [ ] 단계 3: TF-IDF 벡터의 shape가 올바른가? (훈련: (10000, 특성수), 테스트: (3000, 특성수))
- [ ] 단계 4: 모델이 학습되고 정확도가 출력되는가?
- [ ] 단계 5: 키워드가 의미 있게 추출되는가?

---

## 💡 힌트

- 정답 코드는 **참고용**입니다. 직접 코드를 작성해보세요.
- 에러가 발생하면 에러 메시지를 자세히 읽어보세요.
- 각 단계를 완료한 후 결과를 출력하여 확인하세요.
- 모르는 함수가 있으면 공식 문서를 참고하세요.

---

## 📝 제출 사항

1. 완성된 코드 파일 (`.py` 또는 `.ipynb`)
2. 각 단계별 실행 결과 (출력 포함)
3. 최종 정확도 및 혼동 행렬

