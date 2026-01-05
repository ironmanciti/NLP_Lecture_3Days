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

# %% [markdown] id="c387e7ef"
# #  문장 내의 단어들을 임베딩
# - Huggingface 무료 모델 사용
# https://huggingface.co/nlpai-lab/KURE-v1

# %% colab={"base_uri": "https://localhost:8080/", "height": 482, "referenced_widgets": ["faf50a53a990463e890870ad5377096b", "817bf08b3a374eb6bf93a20682e57613", "a3360aec8cff46028aee8f276419197b", "117fcec2ac06480e963a59b06e828730", "843dc93e33dd49d1b528afa8050f0e83", "14ff85326274476097eb322ab9403202", "d513aa7dfc9c463cbe69f914b741616e", "80e7af00253f43fa8390d507719dc6f3", "003db03d565d495aa086845ac7b0c102", "b75cee0cf36341189f71e656075ee82a", "1f4386ea61c040e8ae1b597ea1192634", "8074792f7c3349d2bd586beecbf055ff", "28127866f23240f1a0a3a328404b3a05", "6e385b35df4645a7beff0a2f401b7261", "8e3e52b0913446cabd10e1b7c1d763b2", "b6d34d6770f8486c8eb3904e19d5cb2f", "5992093fb25441dc91961f936d7f314e", "d467d1d8fda8468fb027f4bac661ef71", "d30d9ec97c28460583d8de832c60fdec", "3794f55ba9324b3a815173f2dd179406", "a111224602794e5daa13e110b7284086", "71f90a7a7d7c4bb2950082a40442d4be", "ba116833bad54d28b7b850aa02134c03", "f707d73ddac64291aa8c341e7418b0f4", "478baa5e6d83466eb01c35ce84a72582", "3b30c29db79e460999f1a7b19361a797", "2fd6dcb4ec9041cd9e8850ee181f60f5", "494105e30dbf408997e414258640cd33", "ebebf46a8ad7497287994fd2129ef8c0", "5a55793c8c6d45f19b439b8f171b668e", "86ab50e693404d5489695d98cd04e481", "f52e181a25734185a69dab0799fa553d", "44ae5582702148738bc1bbb5517d9c1e", "37bd97f06f6f4e1fad516f718be69f3f", "309df4b33e9344459fe6eab356ac6033", "1123f799f5a24f49b6fd6ad530a8cc41", "016263e1d274442f89899efcc6b70612", "404851f9ed254865a180933105cb839e", "ff8609ee5ce149dbb3b394b92a9c67c7", "4703e6867a784a1c9cc6e5d0242e3d05", "106bbde8d02c4846b6e3e5ebfa6e4824", "89a2873f0c5844e19f1fdc0a7b9ed227", "85c98d2b96c64550bc9c9321f1505c3a", "c5dcb6fbe7eb45d3bd2c6c44e666c571", "fbe781b830634fa3b9b500e6c15ada60", "c80c4da4a8e242eca2c97ed31b1e35ec", "1519e4eb3f7f4efd8907a7d7a8d6801c", "2684e81c126f4751b26a1f2b5bee2b49", "23fcfadf94eb4df99126b1bf1e2b340a", "2add90d279464b9fab883a270314f59d", "ea3f242b574a412faa31177580aa16d9", "eedd943a3fc141e5973632c738264cdf", "9b9e2bae57c04f10881dbe6a9eeafb96", "aba16a686b2d4ccc8ce4fd3dd1ce37dd", "8bcd9ebbf96f4377b98c5c535f473ea3", "7778ca54248b419081deb15d1feff310", "9b9fddf16428427389ee9b930bfee101", "57b30962985346e7a6fd992f86f2b044", "68d9ae88836c49128540eda67d432937", "d3808da4b9904b68b84d968449ee6edc", "319c8886552b4cf297c68363f275d504", "7163e6808679435697e93b22ae8322fd", "541fc5116d8341fd9391821c26849b99", "6f9d749ce52d4333b896641472fba1c6", "fc14759621264e738a23914a7d742550", "93118f70bcb34352af928d3aaf5b6f7d", "df32415f7ed24ba28d118aa6600a9363", "938ecd331e134b20bc0baa2e19e8e0ae", "3388e52ac7cf4f81835aa19568ff955a", "98a60c11631e4dd8bb7affcd88f43ac1", "1ea51afadc264049aee8ff556a397ad6", "48f857f8b91d4217a3ca96ca1b1bcb82", "2868656fc5874ad3a45732a988324ea8", "440ee24765dc4886a13455671b99527b", "f640570a19544ff6ad8c9f1633a29a23", "5034ef431ca94e799cef3689fe8eb533", "da129f0f5a504072b617762b41b65a81", "266c4756f773414d9b32c696ee6940e0", "a3fc5e57cfbc4a40ba3cfb7c9b7144f7", "f8420b02cf0045828dc3391f2a54e9c1", "68b8eeb5c7d34ff88bb87e594345bddf", "876136e1b6284024bdfa3ae9858337c8", "5266797eefed4bebb1710d7e12ccab6a", "b03fed9d8b4c484bb5dcdee6f6b7dc42", "8b757d65833c4ba49704596c8ad6210b", "84ffdabf7b8346e7abf66abe8edb048a", "e40fd52851654124b01bd377bfed983f", "622026a8307e47dda71f05ec717b3a1f", "ce40283a20f343ac86e4706c650dbb24", "dc27aac7be074d2d9d643cb400ff4e85", "c2fda760c75746bbb1ef2c0c50f151bf", "e0b6e9a9d33b4439a6719d953544140e", "c3d0d33dce0e425da0cdf286a30f2791", "270b2bf4d8b04db98f484a328c7184ad", "9642a2a2a1ca4067900c78d512636357", "214b74b991114a98a5e0138393c50ec9", "8eeeaf816c4845e2b49eedba3bd16901", "257f87d9aa6a41a28677ea68a7ca430e", "c4ea3d5503b847f096f4937d51108f72", "3b796739634f47bf92d1d5a7da981e12", "8df62e2f3fd64011972480ec3dee0913", "8ee5898351594ebb9f392191315a9641", "b233892a4a0349aca1010e9a2d8f5da3", "181f944cb74f44a1b37c5a212b5e5bab", "4c6836a2f38a4c2889216ead7a24790e", "c0e6e18ec182428483c4b874274af80a", "c320bb8f3a574c418963b982f34af8ff", "a11e11cc22f54cce9e259b8003eed74b", "c52cc84570c7487a968f355b9998fb85", "9b2f8ee7a9e149a892d0c09726b3bd84"]} id="8056b154-4ce0-428c-8ce5-dd4189c8e886" outputId="0b39bca7-8cba-466c-8349-3853dba56cdf"
from sentence_transformers import SentenceTransformer

# Hugging Face Hub에서 한국어 임베딩 모델 다운로드
# KURE-v1은 한국어에 특화된 문장 임베딩 모델입니다
model = SentenceTransformer("nlpai-lab/KURE-v1")

# %% colab={"base_uri": "https://localhost:8080/"} id="5bb85f7c-1391-4fcc-b9f2-72120ea085b2" outputId="75b7c04b-434f-4e43-fd93-17998112a528"
# 샘플 데이터: 간단한 문장들의 모음
sentences = [
    '나는 인공지능 공부를 좋아한다.',
    '인공지능은 매우 흥미롭다.',
    '오늘 날씨가 흐리고 비가 온다.'
]

# 문장들을 벡터로 변환 (임베딩 생성)
# encode() 메서드는 각 문장을 고정 길이의 숫자 벡터로 변환합니다
embeddings = model.encode(sentences)

# 임베딩의 형태(shape) 출력
# 결과: (문장 개수, 임베딩 차원) 형태로 출력됨
print(embeddings.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="1c004a7e-2eff-413c-bcec-3a5c4a75b1b1" outputId="52481c77-4e74-4371-e66d-4902baf16bce"
# 임베딩 결과 확인
print('첫 번째 문장의 임베딩 벡터 (처음 10개 값):')
print(embeddings[0][:10])

# %% colab={"base_uri": "https://localhost:8080/"} id="b06381f8-38f0-454a-b599-eefcbd045634" outputId="28ca86d5-5429-4664-e5e9-4d2c89367139"
import pandas as pd

# 임베딩 결과를 DataFrame으로 변환하여 시각화
embedding_df = pd.DataFrame(embeddings, index=sentences)
print(embedding_df.head())

# %% colab={"base_uri": "https://localhost:8080/"} id="2b2bd4fd-4995-4969-92ea-272588414fa5" outputId="3eec3406-da08-4c7f-ca40-703e802e003f"
# 임베딩 간의 유사도 점수를 계산합니다
# similarity_matrix는 각 임베딩 벡터 간의 코사인 유사도를 담은 행렬입니다
similarity_matrix = model.similarity(embeddings, embeddings)
print(similarity_matrix)

# %% colab={"base_uri": "https://localhost:8080/"} id="e5d2874d-7898-4abd-b4ff-ba2db0a917f6" outputId="288b5c0a-8393-4120-ed8c-6761cf21e2b9"
# 문장 간 코사인 유사도를 출력합니다
print('\n문장 간 코사인 유사도:')

# 모든 문장 쌍에 대해 유사도를 계산하고 출력
# 중복을 피하기 위해 상삼각 행렬만 순회 (i < j인 경우만)
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        # i번째 문장과 j번째 문장 간의 유사도 값 추출
        similarity = similarity_matrix[i][j]
        # 두 문장과 그들 간의 유사도를 소수점 4자리까지 출력
        print(f'{sentences[i]} vs {sentences[j]}: {similarity:.4f}')

# %% id="17bef885-706e-4345-8ac2-9139d43eefec"
