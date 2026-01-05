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

# %% [markdown] id="title"
# # 602. RAG (Retrieval Augmented Generation) with Gemini API
#
# - Gemini Embedding API를 이용한 문서 임베딩
# - 유사도 검색을 통한 관련 문서 검색
# - Gemini API를 이용한 답변 생성

# %% id="imports"
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 환경 변수 로드
load_dotenv()

# Gemini API 클라이언트 초기화
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# %% [markdown] id="documents_section"
# ## 1. 문서 데이터 준비
#
# 실제 사용 시에는 PDF, 웹사이트, 데이터베이스 등에서 문서를 로드합니다.

# %% id="documents"
# 샘플 문서 데이터 (실제로는 외부 소스에서 로드)
documents = [
    "인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등의 작업을 수행할 수 있도록 하는 기술입니다. 머신러닝과 딥러닝은 AI의 하위 분야로, 대량의 데이터를 통해 패턴을 학습합니다.",
    
    "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 AI의 한 분야입니다. 텍스트 분석, 번역, 감성 분석, 챗봇 등 다양한 응용 분야가 있습니다.",
    
    "Transformer는 2017년 Google에서 제안한 딥러닝 아키텍처로, 어텐션 메커니즘을 핵심으로 합니다. BERT, GPT 등 최신 언어 모델의 기반이 되었습니다.",
    
    "RAG(Retrieval Augmented Generation)는 외부 지식 베이스에서 관련 정보를 검색하여 LLM의 답변을 보강하는 기법입니다. 이를 통해 모델의 최신 정보 접근과 정확도가 향상됩니다.",
    
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다. 임베딩 벡터를 저장하고 유사도 검색에 활용됩니다."
]

print(f"총 {len(documents)}개의 문서가 준비되었습니다.")

# %% [markdown] id="embedding_section"
# ## 2. 문서를 임베딩으로 변환
#
# 각 문서를 Gemini Embedding API를 사용하여 벡터로 변환합니다.

# %% id="embed_function"
def embed_texts(texts, model="gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"):
    """
    텍스트 리스트를 임베딩 벡터로 변환
    
    Args:
        texts: 임베딩할 텍스트 리스트
        model: 사용할 임베딩 모델
        task_type: 작업 유형 ("SEMANTIC_SIMILARITY", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT" 등)
    
    Returns:
        numpy array: 임베딩 벡터 행렬
    """
    result = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    
    # 각 임베딩을 numpy 배열로 변환
    embeddings = np.array([np.array(e.values) for e in result.embeddings])
    
    return embeddings

# %% id="embed_documents"
# 문서 임베딩 생성
document_embeddings = embed_texts(
    documents, 
    task_type="RETRIEVAL_DOCUMENT"  # 문서 임베딩용
)

print(f"문서 임베딩 shape: {document_embeddings.shape}")
print(f"각 문서는 {document_embeddings.shape[1]}차원 벡터로 표현됩니다.")

# %% [markdown] id="search_section"
# ## 3. 쿼리 임베딩 및 유사도 검색
#
# 사용자 쿼리를 임베딩으로 변환하고, 문서 임베딩과의 유사도를 계산합니다.

# %% id="search_function"
def search_relevant_documents(query, document_embeddings, documents, top_k=3):
    """
    쿼리와 가장 유사한 문서를 검색
    
    Args:
        query: 사용자 쿼리
        document_embeddings: 문서 임베딩 행렬
        documents: 원본 문서 리스트
        top_k: 반환할 상위 문서 개수
    
    Returns:
        list: (유사도, 문서) 튜플 리스트
    """
    # 쿼리 임베딩 생성
    query_embedding = embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY"  # 쿼리 임베딩용
    )[0]
    
    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # 상위 k개 문서 선택
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 결과 반환
    results = []
    for idx in top_indices:
        results.append((similarities[idx], documents[idx]))
    
    return results

# %% [markdown] id="rag_section"
# ## 4. RAG 시스템 구현
#
# 검색된 문서를 컨텍스트로 사용하여 Gemini API로 답변을 생성합니다.

# %% id="rag_function"
def rag_query(query, documents, document_embeddings, top_k=3):
    """
    RAG를 사용하여 쿼리에 대한 답변 생성
    
    Args:
        query: 사용자 쿼리
        documents: 문서 리스트
        document_embeddings: 문서 임베딩 행렬
        top_k: 검색할 상위 문서 개수
    
    Returns:
        tuple: (생성된 답변, 관련 문서 리스트)
    """
    # 1. 관련 문서 검색
    relevant_docs = search_relevant_documents(
        query, document_embeddings, documents, top_k
    )
    
    # 2. 컨텍스트 구성
    context = "\n\n".join([
        f"[문서 {i+1}] {doc}" 
        for i, (score, doc) in enumerate(relevant_docs)
    ])
    
    # 3. 프롬프트 구성
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변:"""
    
    # 4. Gemini API로 답변 생성
    model = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return model.text, relevant_docs

# %% [markdown] id="test_section"
# ## 5. RAG 시스템 테스트

# %% id="test_queries"
# 테스트 쿼리들
test_queries = [
    "인공지능이란 무엇인가요?",
    "RAG는 어떻게 작동하나요?",
    "Transformer 모델에 대해 설명해주세요.",
    "벡터 데이터베이스는 무엇인가요?"
]

# %% id="run_rag"
for query in test_queries:
    print("=" * 80)
    print(f"질문: {query}")
    print("-" * 80)
    
    # RAG로 답변 생성
    answer, relevant_docs = rag_query(query, documents, document_embeddings, top_k=2)
    
    # 검색된 문서 출력
    print("\n[참고 문서]")
    for i, (score, doc) in enumerate(relevant_docs, 1):
        print(f"{i}. (유사도: {score:.4f}) {doc[:100]}...")
    
    # 생성된 답변 출력
    print(f"\n[답변]")
    print(answer)
    print()

# %% [markdown] id="visualization_section"
# ## 6. 유사도 검색 시각화

# %% id="similarity_visualization"
def visualize_similarities(query, documents, document_embeddings):
    """
    쿼리와 문서들 간의 유사도를 시각화
    """
    # 쿼리 임베딩
    query_embedding = embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY"
    )[0]
    
    # 유사도 계산
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(documents)), similarities)
    plt.yticks(range(len(documents)), [f"문서 {i+1}" for i in range(len(documents))])
    plt.xlabel("코사인 유사도")
    plt.title(f"쿼리: '{query}'")
    plt.tight_layout()
    plt.show()
    
    return similarities

# %% id="visualize_example"
# 예시 시각화
similarities = visualize_similarities(
    "인공지능과 머신러닝의 차이는?",
    documents,
    document_embeddings
)

# %% [markdown] id="chunking_section"
# ## 7. 실제 문서 로드 예제 (선택사항)
#
# 실제 파일에서 문서를 로드하는 예제입니다.

# %% id="load_documents"
def load_documents_from_text(text, chunk_size=500, overlap=50):
    """
    긴 텍스트를 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 크기 (문자 수)
        overlap: 청크 간 겹치는 부분 (문자 수)
    
    Returns:
        list: 청크 리스트
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# %% id="chunking_example"
# 예시: 긴 텍스트를 청크로 분할
long_text = """
인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등의 작업을 수행할 수 있도록 하는 기술입니다.
머신러닝은 AI의 하위 분야로, 명시적인 프로그래밍 없이 데이터로부터 학습하는 능력을 컴퓨터에 부여합니다.
딥러닝은 머신러닝의 한 분야로, 인공 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습합니다.
자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 AI의 한 분야입니다.
"""

chunks = load_documents_from_text(long_text, chunk_size=100, overlap=20)
print(f"텍스트가 {len(chunks)}개의 청크로 분할되었습니다.")
for i, chunk in enumerate(chunks, 1):
    print(f"\n청크 {i}: {chunk}")

# %% [markdown] id="enhanced_section"
# ## 8. 개선된 RAG 시스템 (메타데이터 포함)
#
# 문서에 메타데이터를 추가하여 더 정확한 검색이 가능하도록 개선합니다.

# %% id="document_class"
class Document:
    """문서와 메타데이터를 포함하는 클래스"""
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self):
        return self.content

# %% id="create_documents"
# 문서 객체 생성
document_objects = [
    Document(
        documents[0],
        {"category": "AI 기본", "source": "교재 1장"}
    ),
    Document(
        documents[1],
        {"category": "NLP", "source": "교재 2장"}
    ),
    Document(
        documents[2],
        {"category": "딥러닝", "source": "교재 3장"}
    ),
    Document(
        documents[3],
        {"category": "RAG", "source": "교재 4장"}
    ),
    Document(
        documents[4],
        {"category": "데이터베이스", "source": "교재 5장"}
    ),
]

# %% id="enhanced_embeddings"
# 문서 내용만 추출하여 임베딩
document_contents = [str(doc) for doc in document_objects]
document_embeddings_enhanced = embed_texts(
    document_contents,
    task_type="RETRIEVAL_DOCUMENT"
)

# %% id="enhanced_rag_function"
def enhanced_rag_query(query, document_objects, document_embeddings, top_k=3):
    """
    메타데이터를 포함한 개선된 RAG 쿼리
    """
    # 검색
    document_contents = [str(doc) for doc in document_objects]
    relevant_docs = search_relevant_documents(
        query, document_embeddings, document_contents, top_k
    )
    
    # 메타데이터와 함께 컨텍스트 구성
    context_parts = []
    for i, (score, content) in enumerate(relevant_docs):
        # 해당 문서의 메타데이터 찾기
        doc_obj = next(
            (d for d in document_objects if d.content == content),
            None
        )
        metadata_str = ""
        if doc_obj and doc_obj.metadata:
            metadata_str = f" (카테고리: {doc_obj.metadata.get('category', 'N/A')}, 출처: {doc_obj.metadata.get('source', 'N/A')})"
        
        context_parts.append(f"[문서 {i+1}]{metadata_str}\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # 프롬프트 구성
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변:"""
    
    # 답변 생성
    model = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return model.text, relevant_docs

# %% id="test_enhanced_rag"
# 개선된 RAG 테스트
print("=" * 80)
print("개선된 RAG 시스템 테스트")
print("=" * 80)

answer, relevant_docs = enhanced_rag_query(
    "RAG 시스템은 어떻게 작동하나요?",
    document_objects,
    document_embeddings_enhanced,
    top_k=2
)

print(f"\n[답변]\n{answer}")

# %% id="end"

