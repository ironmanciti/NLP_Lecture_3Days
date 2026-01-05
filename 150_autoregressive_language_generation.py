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

# %% [markdown] id="ae55c84f"
# # Autoregressive (자동회귀) 문장 생성
#
# - Colab 에서 실행
#
# 간단한 자동 회귀적인 텍스트 생성을 위한 코드 예제입니다. 여기서는 네이버의 HyperCLOVAX 모델을 사용합니다. HyperCLOVAX는 한국어에 특화된 대화형 언어 모델로, 시스템 메시지와 사용자 메시지를 구분하여 입력받는 구조를 가지고 있습니다. 이는 GPT 계열 모델과 유사한 자동회귀 구조를 가지고 있으므로, 동일한 접근 방식을 사용할 수 있습니다.

# %% colab={"base_uri": "https://localhost:8080/", "height": 304, "referenced_widgets": ["d6ae648605d8434dbe64950a9533b78a", "ef4a6856911d46d2a29842ef9de3c871", "3654f3b06d414ebd8378359150383ada", "936b498feb634bc68174a4d2a99834d5", "e30e828809244ccdb8a8606d015705e2", "541748ccc47145618346975f474bd1cd", "e3b4e85c7ab2477f98da6912e9a21792", "15c655e9860749fe9479ac3deb685264", "7f3559fa09b442519239f227af0ae5fc", "2e3b96c51d9a41b099200549e2c8fa67", "e3ceaf24a1c64fb3910b7f74c56c31fc", "b1635221ac3f4481a2004a5cc8a5c54a", "4d0c07d6c35b4347bed46cb1a70d8b74", "4f4002a07b17461cbeadfedd7aeb056a", "64546bf3318f434386fbc2ad8fc146be", "6a6ec1e929c04949bc9f5be6348ba51d", "f20e9224909e46c79cb8e798ba6bb6a3", "e8011bd115a24173bc00a530fbfd9015", "c864cdb322ce4ee7a723beb1e945bc30", "a879bd738da342a89687cbfaa5098e39", "7627b8fa808d4874a6d9fe797802e831", "f76b00c099244abea1a56fbcf202d722", "4a43db3c2efe49e1af111796943e2f0c", "56ea9ab049dc4d319cf33c512c5302cb", "70dcbceaf9544b12b3cae0c2601f58f4", "46dcf0a5fbb74e478ba391c60b68d052", "447c4a37ffb449878a8c154834465a8e", "da0f3ea91d5749aba095181fb0b2c0f2", "3d5c3265de01443a8a383cef147e3c65", "3433a29214ae4cd882f73f5acbe72be9", "2c9864a2cb8341a78381f3ea06f39b0c", "81d36e79bc304185b49a7a29b72d83b1", "5ca3652cda584373a8c94030e24b5d49", "053cba7ab25c46fb83e320ee83a31065", "abdebbfc842e469baf12ab9444d802f8", "1ec712f295204b369b886cf7c8bf05c1", "622a9dc53f8045509586160b5896f7e7", "7d8817687d7d478ea69beb7fc0bb5961", "1bd9718252d3414d945467c9c2949813", "cd9f8f58f17b4b54a1a93878bcaefb3c", "5fa569e46d9e42d1af55db8f28828f16", "88625f6897d04e65ae33ff2c81e07ddf", "e33f24eeb962486480acd0b5182e853d", "1d4a7f69d544477ab20398d053bd7907", "462b13faf44a45d6887da57f41d62735", "d3e80e0a08664811b928512e752266ca", "42e4c40c780047789388f1c3ce696816", "af8a9b6fb9c6432ea9617df9109ec046", "bfdc45cd0c9e414ba9e300b43b3536fd", "a51bedf0058a4f0abc7d657f5d848d24", "8df6b98bc52c432bb3d0004b000e1bdc", "176a21ca55b44286bcaea3c4a02aca8e", "25265f90668e49a69f58203abef3e6be", "86dbc155c88c4e11a0742fe8e7f025a8", "f34662657ee94390af7f800b9fb762fd"]} id="aBc5bQic1YfQ" outputId="8abb7cfd-4bcc-40bc-8cb6-c22a70f54f9c"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 사용할 사전 학습 모델 이름 (네이버 HyperCLOVA X SEED 시리즈 중 하나)
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

# 사전 학습된 언어 모델 로드
# AutoModelForCausalLM: "Causal Language Modeling" (다음 단어 예측)용 모델 클래스
# device_map="auto" → GPU가 있으면 자동으로 GPU에 올려줌 (없으면 CPU 사용)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 해당 모델과 호환되는 토크나이저 로드
# 토크나이저: 문장을 토큰 단위로 분해하고 숫자 ID로 변환하거나 다시 문자열로 복원하는 역할
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% colab={"base_uri": "https://localhost:8080/"} id="hNNFKA4L31HS" outputId="b98833c6-f1ae-4b34-d751-df71c95125b6"
# 대화 시작 부분 설정
system_content = ""  # 시스템 메시지 (여기서는 비워둠)
user_content = "옛날 옛적에"  # 사용자 입력 문장

# chat 형식으로 대화 내용 구성
# 'role'은 대화 참여자 구분 (system / user / assistant)
chat = [
    {"role": "user", "content": user_content},
]

# chat 템플릿을 모델 입력 형식으로 변환
# add_generation_prompt=True → 모델이 이후 이어서 문장을 생성하도록 프롬프트 추가
# return_dict=True → dictionary 형태로 반환
# return_tensors="pt" → PyTorch 텐서 형태로 반환
inputs = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# 모델과 동일한 디바이스(GPU/CPU)에 입력 데이터 로드
inputs = inputs.to(model.device)

# 전체 입력 토큰 확인
print("전체 입력 토큰:", inputs['input_ids'])
print("토큰 개수:", inputs['input_ids'].shape[1])  # 입력 시퀀스의 전체 토큰 개수

# 전체 입력 토큰을 다시 디코딩하여 원문 형태로 복원
full_tokens = inputs['input_ids']
decoded_full = tokenizer.decode(full_tokens[0], skip_special_tokens=False)  # special_tokens 포함
print("전체 디코딩 결과 (special_tokens 포함):")
print(decoded_full)

# -------------------------------------------------------------
# 사용자 입력 부분만 별도로 토큰화 및 디코딩
# -------------------------------------------------------------

# 단순히 user_content(문장)만 토큰화하여 확인
simple_tokens = tokenizer.encode(f"{user_content}", return_tensors="pt")

# 토큰화된 결과와 토큰 개수 출력
print(f"{user_content} 토큰:", simple_tokens)
print(f"{user_content} 토큰 개수:", simple_tokens.shape[1])

# 다시 토큰을 텍스트로 복원
decoded = tokenizer.decode(simple_tokens[0], skip_special_tokens=True)
print("디코딩 결과:", decoded)

# %% colab={"base_uri": "https://localhost:8080/"} id="jitAS2f34N55" outputId="57b80ca8-5faa-4d3e-d484-32bb9c3c5753"
# 문장 생성 (모델이 이어서 텍스트를 생성하도록 함)
output_ids = model.generate(
    **inputs,                 # inputs 딕셔너리를 언패킹하여 전달 (input_ids, attention_mask 등 포함)
    max_length=100,           # 생성할 문장의 최대 길이 (토큰 단위)
    num_return_sequences=1,   # 생성할 문장(시퀀스) 개수 (1개만 생성)
    repetition_penalty=1.2,   # 반복 패널티 (값이 클수록 같은 단어 반복을 억제)
    eos_token_id=tokenizer.eos_token_id,  # 문장 종료 토큰 ID (End Of Sentence)
    pad_token_id=tokenizer.eos_token_id   # 패딩 토큰을 EOS 토큰으로 대체 (오류 방지용)
)

# 생성된 출력 토큰 ID 확인
output_ids

# %% colab={"base_uri": "https://localhost:8080/"} id="zVmTwgeQ4WpE" outputId="97d8e08c-5259-46ee-f0a5-a54cd6090901"
# 생성된 문장을 텍스트로 디코딩 (HyperCLOVA-X 방식)
# batch_decode(): 여러 문장을 한 번에 디코딩할 수 있음
# skip_special_tokens=True → <bos>, <eos> 등의 특수 토큰은 제거
output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

# 생성된 결과에 불필요한 종료 토큰(<|endofturn|>, <|stop|>)이 포함되어 있으면 잘라냄
# HyperCLOVA-X 계열 모델은 대화 종료를 나타내는 특수 토큰을 출력할 수 있음
for stop_str in ["<|endofturn|>", "<|stop|>"]:
    if stop_str in output_text:
        output_text = output_text.split(stop_str)[0]  # 해당 구간에서 잘라내기

# 최종 생성 결과 출력
print(f"Generated text: {output_text}")

# %% [markdown] id="c10e2ff5"
# HyperCLOVAX는 자체적으로 autoregressive 모델입니다. "Autoregressive"란, 이전에 생성된 토큰들을 기반으로 다음 토큰을 생성하는 모델을 의미합니다.
#
# 위의 코드에서 `model.generate` 메서드는 이미 autoregressive한 방식으로 문장을 생성합니다. 그러나 이를 명시적으로 보여주기 위해 각 단계에서 토큰을 하나씩 생성하는 autoregressive한 코드를 아래에 작성하겠습니다:

# %% colab={"base_uri": "https://localhost:8080/"} id="7kRgp_2C48PJ" outputId="f3ef769d-2a2e-4555-e80b-fb4a1e064337"
# 사용자 입력 문장 정의
user_content = "옛날 옛적에"   # 모델에 입력할 사용자 문장

# 대화(chat) 형식의 입력 구성
# - role: 대화 참여자의 역할 ("system", "user", "assistant" 중 하나)
# - content: 각 발화의 실제 텍스트
chat = [
    {"role": "user", "content": user_content},
]

# chat 데이터를 모델 입력 형식으로 변환
# apply_chat_template() 함수는 모델이 학습한 대화 템플릿(<|user|>, <|assistant|> 등)을 자동으로 추가
# add_generation_prompt=True → 모델이 이어서 대답을 생성할 수 있도록 마지막에 assistant 프롬프트를 추가
# return_dict=True → 반환 값을 dict 형태로 (예: {'input_ids': ..., 'attention_mask': ...})
# return_tensors="pt" → PyTorch 텐서 형태로 반환 (모델 입력용)
inputs = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# 모델이 사용 중인 디바이스(GPU 또는 CPU)에 입력 텐서를 로드
inputs = inputs.to(model.device)

# 토크나이저 처리 결과(입력 텐서 구조) 출력
# 'input_ids' : 모델이 처리할 토큰 ID 시퀀스
# 'attention_mask' : 실제 토큰과 패딩 구분용 마스크
inputs.items()

# %% colab={"base_uri": "https://localhost:8080/"} id="7ut71Jnk5TnK" outputId="b38d2da4-138e-4363-baa8-d0fa1c538d01"
# 모델 추론(Forward Pass)
# 입력 데이터를 모델에 전달하여 예측값(logits)을 계산
# **inputs → 딕셔너리를 언패킹하여 전달 (input_ids, attention_mask 등 포함)
predictions = model(**inputs)

# 모델의 출력 중 logits(로짓값) 추출
# logits은 각 토큰 위치마다 다음 단어(토큰)가 될 확률의 원시 점수(raw score)
logits = predictions.logits

# 출력 텐서의 차원(shape) 확인
# (batch_size, sequence_length, vocab_size)
# - batch_size: 입력 문장 수
# - sequence_length: 입력 토큰 길이
# - vocab_size: 모델의 어휘 집합 크기 (예: 50,000개 등)
print(logits.shape)

logits

# %% colab={"base_uri": "https://localhost:8080/"} id="IMhQdaVO5re0" outputId="a1e10f17-d13b-4c4d-ac9f-09e671c9e3d7"
# Autoregressive(자가회귀적) 방식으로 문장 생성
# → 모델이 한 번에 한 토큰씩 다음 단어를 예측하면서 문장을 점진적으로 완성하는 방식

max_length = 50  # 최대 생성 길이(토큰 단위)
input_ids_concat = inputs['input_ids'].clone()  # 입력 토큰 복사 (원본 보존)

# 입력 길이가 최대 길이에 도달할 때까지 반복
while input_ids_concat.shape[1] < max_length:
    # 현재까지의 입력 토큰을 모델 입력 형식으로 준비
    model_inputs = {'input_ids': input_ids_concat}

    # attention_mask가 있을 경우, 전체 길이에 맞게 1로 채워서 추가
    # (1은 실제 토큰, 0은 패딩을 의미함)
    if 'attention_mask' in inputs:
        model_inputs['attention_mask'] = torch.ones_like(input_ids_concat)

    # 모델 추론(Forward Pass)
    # 현재까지의 토큰을 입력으로 주고 다음 토큰의 확률분포(logits) 계산
    predictions = model(**model_inputs)
    logits = predictions.logits

    # 가장 마지막 토큰 위치의 logits에서 확률이 가장 높은 토큰 선택
    # torch.argmax(logits[0, -1]) → 마지막 시퀀스의 마지막 토큰에 대한 예측 결과
    predicted_token = torch.argmax(logits[0, -1]).item()
    # print(predicted_token)  # 디버깅용: 예측된 토큰 ID 확인

    # 생성된 토큰을 기존 입력 시퀀스 뒤에 이어붙이기
    input_ids_concat = torch.cat(
        [input_ids_concat, torch.tensor([[predicted_token]], device=input_ids_concat.device)],
        dim=1  # 시퀀스 길이 방향으로 연결
    )

    # 현재까지 생성된 전체 토큰 시퀀스 출력 (디버깅용)
    print(input_ids_concat)

# %% colab={"base_uri": "https://localhost:8080/"} id="573d7029" outputId="b11dd855-1353-417f-95cc-12f7069f0568"
# 생성된 문장을 텍스트로 디코딩
# input_ids_concat[0] : 모델이 생성한 전체 토큰 시퀀스 (1차원 텐서)
# skip_special_tokens=True → <bos>, <eos> 등 특수 토큰은 제거하고 자연어만 복원
decoded_text = tokenizer.decode(input_ids_concat[0], skip_special_tokens=True)

# HyperCLOVA-X 계열 모델은 대화 종료나 중단 지점을 나타내는 특수 토큰을 출력할 수 있음
# 예: "<|endofturn|>", "<|stop|>"
# 이런 토큰이 포함되어 있다면 해당 지점까지만 문장을 남기고 이후는 제거
for stop_str in ["<|endofturn|>", "<|stop|>"]:
    if stop_str in decoded_text:
        decoded_text = decoded_text.split(stop_str)[0]  # 해당 문자열을 기준으로 앞부분만 남김

# 최종 생성된 문장 출력
print(decoded_text)

# %% id="O3E_DXy0Na-0"
