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

# %% [markdown] id="f609bd0c-79ec-4ce3-bfb0-4d17006a51e6"
# # Text 생성 및 Prompt 예제
#
# HyperCLOVAX‑SEED‑Text‑Instruct‑0.5B는 지시문 기반 텍스트-투-텍스트 모델로, 한국어 언어 및 문화 이해에 뛰어난 성능을 보입니다. 유사한 규모의 외부 경쟁 모델과 비교했을 때 수학적 성능이 향상되고, 한국어 능력이 크게 개선되었습니다. 이 모델은 HyperCLOVAX 시리즈에서 현재 출시된 모델 중 가장 작은 모델이며, 에지 디바이스와 같은 리소스가 제한된 환경에 적합한 경량 솔루션입니다. 최대 4K 토큰의 컨텍스트 길이를 지원하며, 다양한 작업에 적용 가능한 다목적 소형 모델입니다.
#
# - Hugging Face 로그인 및 토큰 생성
#
# https://huggingface.co/settings/tokens 에서 Read 권한의 토큰을 생성하세요.
#
# - 모델 접근 권한 요청
#
# 모델 페이지: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
#
# "Access repository" 버튼을 클릭해서 권한 요청

# %% id="mtjdIF7oYHg2"
from huggingface_hub import login

# 토큰 입력 (직접 입력하거나 환경변수/Colab secret 사용 가능)
login(token="")

# %% colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["6238aebcd44d4ee58271ef0933e3cfeb", "3487acb4f368495aa11248c1a45110cd", "fe3eb01d3b464d4fad51f5fae67fe261", "65ec9a9056bc41c59a804856748e616b", "16c6f2c1ad714b169bc5649026adac1f", "784ea86c4da44dff8176e745c1d7ec5e", "fb6a9a98ecf94b4cbdeba3b536d1042f", "4499daf2aa8045e6ac5b3ea23f855b6c", "bfebf2f02fa44a1e906d4e210c24cf5e", "43e0a844ccde4e62a0232bda1b86b0b2", "dd53510ede8049deaf4452d08e18ac15", "7f075ab5269848ac9d2c03ef6f5db31c", "3b8fbdc5efcf4387b533fa91ebf09397", "a1df0e606200459f9e67886e5268ef90", "0e0ef212d51647a18c180522159b0e69", "356d9e1095c14cab8f3906abc435d5fb", "68b14abba4cb4e12b75e65978aa29111", "3bc448fdedc644e0bef8ca48e545f730", "f3d9d7cb887e4baf8db7f4bacba9e052", "8d74468a73bd4df9894fb717d4a52fde", "26b78a3388134643b3b1d508364159ec", "bad4e5a8bcd045139dcf7a93c5ef2275", "5c9de5c9d4704b2797ebee9efcd3a30a", "ff22727f974f49bc97bdb6870ee8e6ca", "fb438fa834fc46d3828343745010298c", "48518d216caf48b39a1e05601c00f04b", "8173ce7278ea4efcbcfde481a7f056c9", "b7280853822f4da99235a6f463667952", "845171292f994c938014bb2771c6ef6c", "995ffa2bc1f34c47bdd3f686dab237f4", "19186cd28dce439e99cf14372260c83a", "29f3891f4266487294be210b947eacc7", "c2873a7c87504624bac2d0d477b03c2b", "b85da169913743288e8ccdbed3e5858f", "251383b320ab413790422a7caeb0b431", "fe2a79a288a548c8a0488139c132fc66", "2feebb47b58c49158f5d132512b3065a", "d7b2446e111541e9a5f543138c701278", "a4b09401139049d6a9cf6b5fbbd1e446", "f5203a83eaba42cda73c0580e0146d97", "4b141f26505f4a3282b62b7022e9205b", "aa2938926fc3452394fc81fd85a2948f", "f5df260201964a4e92aa718c5962fc96", "9a0efe8de72f44babf706d09af64fd1d", "5f13b346ac564a1fa9f1452b93c6cc45", "52718fbe37604a73b2eaa28e940264a7", "81299cdbe75a4721a638b9ff24701b6b", "ceda3475b5c04d41aa790b8745fddeb5", "0b79cd01940240619b2c5f59547a7c51", "f1d190ebe7474812859b7d161e962c27", "4cf989f1c00742c2baec284efc6a6e89", "250ae204aa6e452b9ef2806111d80aa9", "f498b3f6cb2543a68346c1bdb1b2478f", "c87f441ed1b246959d6e6caccd83d197", "d6c0eb0e0be543df85d9b2d690ebf67a"]} id="c78aaec8-fbb7-454f-b2b1-8de83b139288" outputId="ef55904d-78ac-409f-9ee8-20e2d0c2b8cd"
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %% [markdown] id="6e00626d-ca14-411b-898f-856309a98b6d"
# ## 문법 수정
#
# SYSTEM : 당신은 문장을 받게 될 것이며, 당신의 임무는 그것을 표준 한국어로 변환하는 것입니다.  
# USER :   
# 안갔어 시장에 그녀는.

# %% id="tFi0VaHuALm5"
def generate_response(system_content, user_content, max_length=1024, repetition_penalty=1.2):
    """
    HyperCLOVAX 모델을 사용하여 응답을 생성하는 함수

    Args:
        system_content (str): 시스템 메시지 내용
        user_content (str): 사용자 메시지 내용
        max_length (int): 최대 생성 길이 (기본값: 1024)
        repetition_penalty (float): 반복 패널티 (기본값: 1.2)

    Returns:
        str: 생성된 응답 텍스트
    """
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 필요시 <|endofturn|>, <|stop|> 등에서 자르기
    for stop_str in ["<|endofturn|>", "<|stop|>"]:
        if stop_str in output_text:
            output_text = output_text.split(stop_str)[0]

    return output_text


# %% colab={"base_uri": "https://localhost:8080/"} id="m0tb3t-P_Kxp" outputId="bfbb5574-43d5-41c3-d56d-c7630cd528fb"
# 문법 수정 예시
system_msg = "당신은 문장을 받게 될 것이며, 당신의 임무는 그것을 표준 한국어로 변환하는 것입니다."
user_msg = "안갔어 시장에 그녀는."

result = generate_response(system_msg, user_msg)
print(result)

# %% [markdown] id="54f4415f-60b6-4308-aaf1-9f46dc634445"
# ## 구조화되지 않은 데이터의 구문 분석
# SYSTEM : 구조화되지 않은 데이터가 제공되며 이를 CSV 형식으로 구문 분석하는 작업이 수행됩니다.  
# USER :   
# 최근 발견된 행성 구크럭스(Goocrux)에서는 많은 과일이 발견됐다. 그곳에서 자라는 네오스키즐이 있는데, 보라색이고 사탕 맛이 납니다. 회색 빛이 도는 파란색 과일이고 매우 시큼하며 레몬과 약간 비슷한 로헤클(loheckles)도 있습니다. 포유닛은 밝은 녹색을 띠며 단맛보다 풍미가 더 좋습니다. 네온 핑크색 맛과 솜사탕 같은 맛이 나는 루프노바도 많이 있습니다. 마지막으로 글로울(glowls)이라는 과일이 있는데, 이 과일은 신맛과 부식성이 있는 매우 신맛과 쓴맛이 나며 옅은 오렌지색을 띠고 있습니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="pRK-bu-K_760" outputId="17d06fcf-e984-484e-86a9-aabb4d10dc27"
system_msg = "구조화되지 않은 데이터가 제공되며 이를 CSV 형식으로 구문 분석하는 작업이 수행됩니다."

user_msg = """
최근 발견된 행성 구크럭스(Goocrux)에서는 많은 과일이 발견됐다. 그곳에서 자라는 네오스키즐이 있는데, 보라색이고 사탕 맛이 납니다.
회색 빛이 도는 파란색 과일이고 매우 시큼하며 레몬과 약간 비슷한 로헤클(loheckles)도 있습니다. 포유닛은 밝은 녹색을 띠며 단맛보다 풍미가 더 좋습니다.
네온 핑크색 맛과 솜사탕 같은 맛이 나는 루프노바도 많이 있습니다. 마지막으로 글로울(glowls)이라는 과일이 있는데,
이 과일은 신맛과 부식성이 있는 매우 신맛과 쓴맛이 나며 옅은 오렌지색을 띠고 있습니다.
이 데이터를 과일명, 색상, 맛으로 분석해 주세요.
"""

result = generate_response(system_msg, user_msg)
print(result)

# %% [markdown] id="05e0d9c0-12f4-4ed8-aaf8-a704505fcb98"
# ## Keyword 추출

# %% colab={"base_uri": "https://localhost:8080/"} id="c99d0449-9850-4e25-b779-2e1041a1577a" outputId="ec1c2d03-a8e8-4d31-d1c7-f85365d98dfe"
text = """
"블랙 온 블랙 도자기(Black-on-Black ware)는 뉴멕시코 북부의 푸에블로 원주민 도자기 예술가들이 개발한 20세기 및 21세기 도자기 전통입니다.
전통적인 환원 소성 블랙웨어는 푸에블로 예술가들에 의해 수세기 동안 만들어졌습니다.
지난 세기의 흑색 자기는 표면이 매끄럽고 선택적 버니싱이나 내화 슬립을 적용하여 디자인을 적용한 제품입니다.
또 다른 스타일은 디자인을 조각하거나 절개하고 융기된 부분을 선택적으로 연마하는 것입니다.
여러 세대에 걸쳐 Kha'po Owingeh와 P'ohwhóge Owingeh 푸에블로의 여러 가족은 여주인 도예가들로부터 전수받은 기술을 사용하여 검은 바탕에 검은 도자기를 만들어 왔습니다.
다른 푸에블로 출신의 예술가들도 검정색 바탕에 검정색 도자기를 제작했습니다. 몇몇 현대 예술가들은 조상의 도자기를 기리는 작품을 만들었습니다."
"""

instructions="텍스트 블록이 제공되며, 당신의 임무는 텍스트 블록에서 키워드 목록을 추출하는 것입니다."

result = generate_response(instructions, text)
print(result)

# %% [markdown] id="a1934098-ba94-4cce-a64b-0a46566f67c2"
# ## 감성 분류기
# - 한개의 text 감성 분석

 # %% colab={"base_uri": "https://localhost:8080/"} id="fzHN82g9Fptf" outputId="a65b3052-46b4-4092-d201-a682b1bdd68c"
 instruction="당신은 텍스트를 입력 받게 될 것이고, 당신의 임무는 텍스트의 감정을 긍정적, 중립적, 부정적으로 분류하는 것입니다."
 input="나는 새로운 배트맨 영화가 좋습니다!"

result = generate_response(instruction,  input)
print(result)

# %% [markdown] id="55aee066-e4e5-4e29-894e-8519710a44c5"
# ## 회의록 요약
#
# SYSTEM : 회의록이 제공되며 귀하의 임무는 다음과 같이 회의를 요약하는 것입니다.  
#
#  -토론의 전반적인 요약  
#  -행동항목(무엇을 해야 하는지, 누가 하는지)  
#  -해당하는 경우 다음 회의에서 더 자세히 논의해야 할 주제 목록입니다.  

# %% colab={"base_uri": "https://localhost:8080/"} id="fd249722-e848-4ce9-8e20-84103c525c4d" outputId="9b940ebc-46f5-426c-a175-13082c613001"
meeting_minutes = """
회의 날짜: 2050년 3월 5일
 미팅 시간: 오후 2시
 위치: 은하계 본부 회의실 3B

 참석자:
 - 캡틴 스타더스트
 - 퀘이사 박사
 - 레이디 네뷸라
 - 초신성 경
 - 혜성 씨

 오후 2시 5분에 캡틴 스타더스트가 회의를 소집했습니다.

 1. 새로운 팀원인 Ms. Comet에 대한 소개와 환영 인사

 2. Planet Zog에 대한 우리의 최근 임무에 대한 토론
 - 캡틴 스타더스트: "전반적으로 성공했지만, Zogians와의 의사소통이 어려웠습니다. 언어 능력을 향상시켜야 합니다."
 - 퀘이사 박사: "동의합니다. 즉시 Zogian-영어 사전 작업을 시작하겠습니다."
 - Lady Nebula: "Zogian 음식은 말 그대로 이 세상의 것이 아니었습니다! 우리는 배에서 Zogian 음식의 밤을 갖는 것을 고려해야 합니다."

 3. 7구역 우주 해적 문제 해결
 - 초신성 경: "이 해적들을 처리하려면 더 나은 전략이 필요합니다. 그들은 이번 달에 이미 세 척의 화물선을 약탈했습니다."
 - 스타더스트 선장: "그 지역의 순찰을 늘리는 것에 대해 스타빔 제독과 이야기하겠습니다.
 - 퀘이사 박사: "저는 우리 함선이 해적의 발각을 피하는 데 도움이 될 수 있는 새로운 은폐 기술을 연구하고 있습니다. 프로토타입을 완성하려면 몇 주가 더 필요할 것입니다."

 4. 연례 은하계 베이크오프 검토
 - Lady Nebula: "우리 팀이 대회에서 2위를 했다는 소식을 전해드리게 되어 기쁩니다! 우리 화성 머드 파이가 대박을 쳤어요!"
 - 혜성 씨: "내년에는 1위를 목표로 합시다. 제 생각에는 승자가 될 수 있을 것 같은 주피터 젤로의 비법이 있습니다."

 5. 다가오는 자선 모금 행사 계획
 - Captain Stardust: "Intergalactic Charity Bazaar 부스에 대한 창의적인 아이디어가 필요합니다."
 - Sir Supernova: "'Dunk the Alien' 게임은 어때요? 외계인 복장을 한 자원봉사자에게 사람들이 물 풍선을 던지게 할 수 있어요."
 - 퀘이사 박사: "승자에게 상금을 주는 '별 이름을 지어라' 퀴즈 게임을 준비할 수 있어요."
 - Lady Nebula: "좋은 아이디어입니다, 여러분. 이제 보급품을 모으고 게임을 준비합시다."

 6. 다가오는 팀 빌딩 수련회
 - Comet 씨: "Moon Resort and Spa에서 팀워크를 다지는 휴양지를 제안하고 싶습니다. 최근 임무를 마친 후 유대감을 형성하고 휴식을 취할 수 있는 좋은 기회입니다."
 - 캡틴 스타더스트: "환상적인 생각이군요. 예산을 확인해 보고 실현할 수 있는지 알아보겠습니다."

 7. 차기회의 안건
 - Zogian-English 사전 업데이트 (Dr. Quasar)
 - 클로킹 기술 진행 보고서(퀘이사 박사)
 - 7번 구역 순찰 강화 결과(캡틴 스타더스트)
 - 은하계 자선 바자회 최종 준비(전체)

 회의가 오후 3시 15분에 연기되었습니다. 다음 회의는 2050년 3월 19일 오후 2시에 은하계 본부 회의실 3B에서 열릴 예정입니다.
"""
instruction = """
          회의록이 제공되며 귀하의 임무는 다음과 같이 회의를 요약하는 것입니다.
             -토론의 전반적인 요약
             -행동항목(무엇을 해야 하는지, 누가 하는지)
             -해당하는 경우 다음 회의에서 더 자세히 논의해야 할 주제 목록입니다.
      """
result = generate_response(instruction, meeting_minutes)
print(result)

# %% [markdown] id="0e064629-07fc-4c86-bcf8-05d1bcdcb6da"
# ## 이모티콘 번역
# SYSTEM : 텍스트가 제공되며, 귀하의 임무는 이를 이모티콘으로 번역하는 것입니다. 일반 텍스트를 사용하지 마십시오. 이모티콘만으로 최선을 다하세요.  
# USER : 인공지능은 큰 가능성을 지닌 기술이다.

# %% colab={"base_uri": "https://localhost:8080/"} id="K-kRE0KwJ7YM" outputId="17331d1d-14a3-4040-b141-d8fa0eeae72c"
instruction = "텍스트가 제공되며, 귀하의 임무는 이를 이모티콘으로 번역하는 것입니다. 일반 텍스트를 사용하지 마십시오. 이모티콘만으로 최선을 다하세요."
content = "인공지능은 큰 가능성을 지닌 기술이다."

result = generate_response(instruction, content)
print(result)

# %% [markdown] id="22cd2c83-3e39-456a-a45d-56ca3fcf3bd5"
# ## 번역

# %% colab={"base_uri": "https://localhost:8080/"} id="Lxl18mD7KOsc" outputId="67a64c1f-3777-44e7-d758-41d46b3c306f"
instruction = "당신은 영어로 된 문장을 받게 될 것이고, 당신의 임무는 그것을 한국어와 동시에 일본어로 번역하는 것입니다."
content = "My name is Jane. What is yours?"

result = generate_response(instruction, content)
print(result)

# %% id="70dc3ac6-2df0-43aa-82d2-168378adc031"

# %% [markdown] id="47062606-7e80-4332-9bbc-f48681169d83"
# ## 실습: 위의 Prompt 내용을 수정해 가며 api 실행

# %% id="153ef2b1-9ba7-4da6-963a-77a52b07212c"
