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

# %% id="arT0NBzSePW7"
# !pip install -q transformers datasets
# !pip install -q sentencepiece
# !pip install -q kobert-transformers

# %% id="VfI48nV66Xmh"
from transformers import pipeline

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

# %% colab={"base_uri": "https://localhost:8080/", "height": 489, "referenced_widgets": ["a934dc97311244608385a2a168c99fc5", "36d6c7d0ab6448ca9dcbe5ffd207e19d", "a0fc629a69ac47639e228eead5c2837c", "1b63a76f812e4f23bec6cac6b03d0389", "1d8d39f9641d4487ab05404f2544f552", "98f02509dd5347dc9c52494e42e414e2", "873e9073d9894f89953cdedb0d962fca", "1a6b69e21d8f4569b4c5489bbbd322bf", "6c1ae0c20df94d29a682ad2840e88adc", "e2ffc86665d447bd9a57d860afa3e83c", "c3601ce3efd041db89cb52e135fe282a", "c14c7cac04214c4f80eacbf4775ace89", "9449456dbdbf443c9954af1b0148859e", "368e5839cade4f8a8ae11a413bff14e4", "6a1ca6a5d821432f8a02eac828bd726b", "0f83c5f934a740649da17b5ccee25dc3", "f3ea5286d7f94573983fbf55051a0b88", "42cda6a85ae840928ac9809ec1bdb84a", "92d886a7b733476d84bee03f188c6dac", "d52fe992021f4a1681f02e0bfd556270", "3fcb95b816cd447198c65fa3f2536105", "bfba729d527c4d57af2dc782974089e8", "4905043726af4ecdaf4db955c3d7d1f8", "59963df7fc4b4e6d8115e905b56afd54", "c354f50b368b42a28f76a8b9c2e9a0fb", "73e04e178fab441ab890c4acb565e690", "6bab5081abf346cc9991c6f85924e5e9", "e80e1d85e2214784b0ce1893a2684553", "00e5e9e4db9849f7b35f8a4896c9e1b1", "c065a8cc2acc4e1e95916a22b4d73808", "819f9357e45e407ab59181b9b3e275a3", "55bdc73d3ab54bfa8911020664c7c1f5", "e7ba05b6e3404517abc5dbc652555e3f", "6950d92a6efd4b3c86e098beaacb5a4a", "67efe9e61d7344d2bd3b326530e8d25c", "1bb0a89188854412939dd481294477b1", "ed79f2aa13f544a8aa0991914abf5171", "ad34492d79b843afb2b6809538f62d10", "57df87b61a74459ea680d8f1589bfa3f", "9d6ecc3a052e44139b033ad1f52e49a3", "dfc54a45403b4f6699989f2cd4796200", "bd626d255c804469b7683e28106a330c", "0fae7203a24f4cd98197b2f93c107afa", "00fddc1126b6490bbcaa4ec1a1a6c2e6", "3f6c357e792b4eda87c88c6e388ba2d1", "d7928e605aae40798b5a432f8f9da7ed", "0bd14fca187f4eb0977bc52298d6ee09", "48a25728e3ba40869fc3b6cc75647176", "808d62ac601248d7b92cf21002f4b5f4", "02b619f0849a47b3a08fc40d33cac0f8", "54003dce3900409e94ba987ef5a4ee85", "55123d6ec26548bd80577a86094cba28", "8baa97c1228e4146955fdca4567b7888", "ce0613fdb2e84839a29748fa0608f018", "2379dfb5f3f94ceaae356be63cb1ab33", "0fc16cd5e4b8431fb4f5b4b0e7fadb0b", "9ea0fe35a96a4fdeb3a09b47d01d3b9a", "253ca3178e5e40d5ad2927686d6850db", "f26c10832a2443ecb2e3c16e3d15c472", "8eaec095c23e4faf886dedcfe3bddea7", "60069a31c2154b9eb411b85a45af565b", "bd0401645416432f939859f27e503b47", "231073c24a084aa1b909b04f94a0f6cd", "a2847b8db1e84a29a6a4a7a6a82e269e", "2be2db5fa5b74c3b99790343b3b68129", "d069655a08764b46ae9607aa30a5cc30", "2ea71f9deee840cd92340745064428bb", "390d0ef3680a484394a97f7819cf4a22", "39162ef0221d4c1c9ff32982049ced82", "9cc92203eb814f6f93345f6d9fffc979", "576a0abc4f95452e9bd18f425d064384", "82574083274249bbafbe6d56caf64a43", "ddcab952cc0b46eab960df31ae573a19", "8cc5f43812964e469670c1ed08af11c2", "f3d6968f045b4085ade739d1e8751bd1", "45a6f273943849ccbc6704b18c17b0dc", "6ed670ca2bc148cb9d1dc05f8439c5bb", "ae50b3bd453748fd8a69057c872a52c6", "63e0a7bb80ff46309a6a69f62deb4cbd", "82849acb33c84fa59bc6a97601f38bfc", "ab6d76af717b4d87b6e3e1a573afb67c", "d42808cfa9ca4303a0d377a3c9a4a935", "da51ea9fae4040808f40eeb44b88546e", "4f8625dfd703495a8c3a4914e1c13297", "de5411423b4f4a0bab57c4c85ae03fc6", "93d440f4c4ab4a5cb7aae381734510b8", "4a7aeb4ad2b54784817ed3562fcbb857", "9144208fde7c41b585bca9c1e29c489c"]} id="mveybhY56Xml" outputId="bb383275-ed1a-4bd2-b0d4-64be378da573"
# HuggingFace Transformers의 번역 파이프라인 생성
# 한국어(Korean) → 영어(English) 번역 모델 사용
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

translator("나는 너를 사랑해"), translator("언제나 당신에게 감사함을 느끼고 있습니다.")

# %% [markdown] id="pcvvG9876Xmp"
# ## 한국어 감정분석
#
# - NSMC(Naver Sentiment Movie Corpus) 로 미세 조정된 BERT 다국어 basecase 모델 : https://huggingface.co/sangrimlee/bert-base-multilingual-cased-nsmc

# %% colab={"base_uri": "https://localhost:8080/", "height": 259, "referenced_widgets": ["b6641e842b3b4d02b75d6af698ecb54f", "90f6c5bfa3e24fa8a946407b6c5d4542", "666217ef60f24aeaad7b338d0e4d1afb", "330c6df7f2974f998ed69270f3f43d03", "870520f57e004a41b4fe83c0e933b535", "5bae337701d74c10ba0c7e1720be4178", "7b4ede0a2f094ce095531131f72bbf63", "287835c1e8064cfab507f6d557aceaa6", "2553868932594f359243c7ebbe7cbb41", "be005ecc8a974b75a172cd6eacdb12d0", "c007038e8553436bbf8bd75924108ebf", "300382ab1118470dbd77375933d9688d", "27b0177fd50247c1b7f8acc677509d40", "3e4929de099148c28f16b8ab990a44fc", "298a8f452c0147eabd5abeff9ddad683", "037faf92b8224acfb7d084352e618cb7", "4c801490ba424db8bec3e311a525b782", "f8be5b1158df4b13bf2797150fb5e8de", "aad39f59cd46489f81e5af3c93b88495", "12053a223a2f4669be556516a1641224", "4302b0dd208941aca5116a452df98354", "d7e57d00ff0e4646b8cdbf3a9ac68d87", "3ec35a6ede8b4ed1814c953275eb9822", "b7468a51dc5c4631af24a4575460f22e", "2a83e62faef64cd49a0cf2028da488e8", "a4f202e27d2d42b98697d0e95729bdc5", "bfb5b62bd2ba43508218faa115756f51", "89d98dd61bbc4fd49707138db5a31a46", "3eaccf8d14ff48d7b59c83f02b0f68dc", "38039ec566864bd7877b55cb52a43d7e", "a5d9167be9074946902f6b3a5450851d", "129abc57dca441d8890709678fc53a76", "459ddf66567d4a7ea1e4f195c3f0254a", "646fb812c09d4a08bc642e63837f821e", "6f9d3e4fbad54b2bb9073de6e4c14c8b", "2ce39b7b0fe643a8bd6518faedd1cae7", "1595a2813e75474a8424454ea962f8f9", "46d6f2d33594452abd073dbd372f321f", "b53f9c2997fa42aa90a33d12a5d10d96", "7050fba366d94bf7ac8c77216c94fa1a", "e7b112824b994d87a525c4be45e82e6a", "89a9735a068549948d2aa6c1cdd08e3a", "9d6b044262e243e48808bdb83fe7601d", "482f0d27baae4aa197edfef17c59ee61", "e4efb628af1b46a38664b54ed1d0f539", "81914855e9324150be9a9c4d5fda8816", "e6214064970b4b6c9a2b30befecc0326", "07524b19e7c0431eb96d944f6a68a6e5", "2335aba975e143bcbbe3ca785fe8cee0", "8d8d7c5611e941e69170daf51984b444", "c7cea48d54b74f2195ba186c3895167b", "f5117ec9eb194128b69aad262466b955", "fa3628ee745b4d1592ee9f8f51eebd7c", "87ef8a84f4bf4be596b87f92863f5f66", "ceaa79691c134e44b63a7e8f5114fefd", "bb049869c0d644788edaf349d3aaba49", "8a599c763ad4473fa96765a2c7162996", "2ee929490e7f4401a49bd44e7febb672", "3a45d2f0a9b6420cbcc0cf6b3fadc8be", "5bb965bd30eb4f5ab9f733a772df759a", "c73f88eac7c14ef9936a1cba919dd36c", "ac44140823054b418804a5443f6c6a42", "9889bb2b65734009bab617ddc1628e27", "5e2323315b7b459781652bc93fe9a6ce", "eb36088090fd4c57920fb1f4ded63fd0", "77f78f6e5fd94b278d32029f354faae6", "e5099f39a94b4f59866e945aa662caa6", "32434a90a070438282ec08d32ef4acd0", "0beec70aeb814918a7ae826e3cb80a46", "5fc0a03520f9458087f98f7001419f4a", "e92986dc155746f8abc0f741539376d5", "8c026ce10764404d976e5550ecf732a2", "369ec4300a264afba114f423080229ad", "7f1ed704eacc418cad7db12e469df730", "fd19f77881ea446a8de457eb06906070", "6c13b84a65ab404ca8524e83a3fc4460", "0e62b6a260924adea22fb8db6a6babdb"]} id="lzjlhBlpePXE" outputId="b97fe59e-d774-4fd6-ffd1-b43f7a24b707"
# 한국어 감성 분석 파이프라인 생성
# 'sangrimlee/bert-base-multilingual-cased-nsmc' 모델은
# 네이버 영화 리뷰 데이터셋(NSMC)으로 학습된 다국어 BERT 기반 감성 분류 모델
classifier_ko = pipeline('sentiment-analysis',
                         model="sangrimlee/bert-base-multilingual-cased-nsmc")

# %% colab={"base_uri": "https://localhost:8080/"} id="zke67uvL6Xmq" outputId="e49e92f8-3d7d-475b-b356-ef69fd5438be"
print(classifier_ko("오늘은 정말 즐거운 날이다. 행복하다."))
print(classifier_ko("기분이 꿀꿀해서 술이나 한잔 해야겠다."))

# %% colab={"base_uri": "https://localhost:8080/"} id="K-Y5gF-D6Xmq" outputId="ec9f71f9-54a8-4ed1-b2ac-45528c18a0d3"
classifier_ko([
               "언제나 당신에게 감사함을 느끼고 있습니다.",
               "너한테는 별로 좋은 기억이 없어."
               ])

# %% [markdown] id="qVRSJKEv5zb4"
# - 자동 별점 부여

# %% colab={"base_uri": "https://localhost:8080/", "height": 302, "referenced_widgets": ["64bd2e109beb4ef4aaee7ece1f26a4f1", "e445c3666ee04b16a8ef6f36a1819300", "bd57b9d54c9d4d68b2a9d6d65ffc7499", "62b44404008248f2be50d9e95987b84e", "304142b75a744f2ea056bb34ade35236", "3bc892174e5843cb8031a35cf993ad7a", "017786f26da64363a397acf75392f453", "97434a674f6e4532a893e02d21486a75", "d76fa028118d4e6588bd248c80149da5", "b9c1859cfc484894834151d32c253006", "0ce76399cfb14e769a222d5050d076ad", "0d3d992e28f541a18d0eed4df566cb36", "b157f1f8a12648fa8f08804e71f389cb", "6d8048a928ac496b9830f5e8af1dbea9", "5e4fd3109f474a3682c3bc9e5ef36d51", "9d9319b7600f4709b0e51fd5e40a83a2", "8ff33079a4214fd585b03e60049e5592", "5aa5bd499f55496899a2275f39a8a1d1", "f6ae7e9aebf54a0dbb5fc3be1864e8b5", "a04e530564c54472b7d5497fbff7b9f4", "1c15da5e6af5409284ce3f095f9db3b4", "90382a106214445996faa3d401327ed9", "91d9715b5bc04c47b16b102a86dae451", "ed6ababb35114767a3f317e71a89eaef", "77ec67c1c86845c7bf8bcd799a6f8131", "9970948bca804428b6ee2eb2eae4a940", "a1bb225107304ef191ff0a9bd1595ab2", "4076eab381564f3394d31aab9a23fca3", "7ff7a7c2b8844f67b5054629ca90dbc9", "cde19017f0754a869a66174ed0701a39", "6b72e0a4f1c6475e9a81c9e50cb49c71", "c63bb6087cc94b7ba2b0e5f15232951a", "511e2fb4bfc44c37bb2326aa9ff1a043", "0bf900639f434b5f995c25d3ce539aa8", "369eaa05ba8745938973bd3d66f481ea", "ec1a1b084d7d458a95ae85b7e8cda040", "84ebaeae06a749f8a3e7d8b38e36d14e", "57111a7dfd9949869c792a99b1c877a1", "f758f0c2de054de0a14a36d63f67fc2a", "99f12485a2e24781a00a7d0d6aa57c91", "c7104115bc1a485cb488d14948a40cf1", "6681f6340bbd46c48360ef0e32869421", "85d4a50fee1144918ffd782508c79a91", "04013d5d66bf479585440e1dffefea06", "a74b68c589434ad2ac1f2c3f39332198", "51aad2ac84804e0f85bbbcf37c969a85", "d6673e257ef943ca9c7370934efa41e4", "f777d9e53b0b496e9e365f7d361eff70", "cc8177203d2041439747931db46e1e18", "a67b459165954a16a5096fffbe8e40b4", "1c9eda96da9346eb9d6cade05b5f7383", "fca5ee4b94f54dde9a5fa87aa8a6d95d", "7d322cbc37814177b59db07baaf5a712", "27b057a350414c74b6f8edd31807595b", "fda9ebbd93bb439fb39d2c9ab9c13f6b"]} id="CPeKZ4Wi3IxS" outputId="b70e6ab1-ca12-49ac-c5b7-efdea4714885"
# 다국어 감성 분석 모델 이름 지정
# 'nlptown/bert-base-multilingual-uncased-sentiment'는 영어뿐만 아니라 한국어 등 여러 언어의 감성 분석이 가능한 BERT 기반 모델
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# 감성 분석 파이프라인 생성 (한국어 문장도 지원됨)
kor_classifier = pipeline('sentiment-analysis', model=model_name)

texts = ["다시는 보고 싶지 않은 짜증나는 영화",
            "아주 재미있는 영화",
            "정말 재미없는 영화였다",
            "이 영화 망할거야",
            "이 영화 최고",
            "보통 영화"]

results = kor_classifier(texts)

for i, result in enumerate(results):
    print(f"{texts[i]} --> 별점: {result['label']}, with score: {round(result['score'], 4)}")

# %% [markdown] id="wr-tP0HF4CmP"
# ## Zero Shot Pipeline - 처음 보는 문장의 category 분류

# %% colab={"base_uri": "https://localhost:8080/", "height": 316, "referenced_widgets": ["8cbb1258a18d44c7b321d2c79b5459b4", "c08b7823e7f842aaa526952acce4d617", "db9212f0d2a4455684ddeca5b72e33e5", "b187d320f87846b6993e0dd7b83f618f", "62d4c3a07a104a5285ac474584f9b94f", "8143495723fe44a2810b14b45e3fe0f8", "32838609c7964029a3b09720e2f9ee44", "63292222b86b477a99b088a2fcf92103", "3b26dcb7d9b04bd1a127cd3a543adab3", "6cf2fce0b148401b9242f907ddb953d4", "3a0cc99cc3764c709bc448163d1daf04", "dd2063328d36413e94a1558467b3dcfb", "3989d844ee7044cc932d12c19956a133", "d5c1bcda16df49479398259567e21607", "99979293a65b41038a4e1893f741ec62", "49a0c0b1393e47c79df8fec638d1fcbe", "6c8498a265e74a0cba479e92d10c8c31", "1b94e443bbdd45979c67c33adec32f70", "399ab57eb24c4881ad5007d2e06c79b5", "f54759b98d1b4b9eb2790a17dbf04805", "e021655fac6f42e08e3dde75bcf4c1a3", "18b3f714eeae43978bf3fd85450c3ac2", "a828984fc98a416eb41184cee1bd8039", "2873aa8000cb42a1956afee157e5a044", "fda68bb77eaf479a8647c00ddc4dac2f", "db170c8dadc142d9b736968c4638de39", "5818fb56c43e4cac95fdcd05afe1f7f5", "5524729e35b94da59e9a8e2e85b79565", "d6839cb25a5a443bb6f5c739333d4700", "bd7bca2f5f14402ebf8baf60627dba49", "b6437166ce33461da4d66c79131c4d66", "da4bc7ab6cf44524a8703b636e2ac314", "a07b6ec7e4a542b681a77c8fac1b709a", "70431eb5c6e14b199f77da1971072783", "027111a6b39e4a71a34b989af7b46ce7", "0df8f2a1189544668fcaf01ceefd7f6a", "493c414de8c24a6ab5df26905910c0de", "eefcd105826241b2a2ce52ba780a542c", "10d5a598678a490b9b9baccc4d71cbcd", "410b5fd52cf547fca63d9b3747fa00ac", "0d7450b3c2014000babf19c6202674a5", "ae4bc1621f4b4c4787986be27456ecad", "fccd6bfc11774adb9c74d82bde0d1b37", "28cb428c96be478091be132d7841c653", "57096a9a27fe484bb06361f5e1d92e65", "2ed789a324f8499a87f67ae08e5d8014", "6f4e8ef2825c49c8b8051105afa4b2a4", "097f271c219f412ab409712ed206e709", "c1be80e2f856485581c60fb0a1a8c1f9", "aa67f19478034946898b0b1649691c40", "a99d3dc455cc412d9ba36c9d2681a55b", "40506e7b2a874a8b92071081ecbfb367", "37e9dff13f414fad8fc9ce2699430823", "d805630e960647dc9ffeba066258be07", "70fdbba885644f03861c2ee44104d7d4", "a64b9697692c47379566f1d3d8b7aaff", "99b30efa1dfc4ac09b3ee15599a776bf", "bbe0217ac84f40ea9be9fdef3261e5b4", "066936fd70dc4c3f9aa2cefb7c504393", "4b32cd6242bc4946bff011292a94a6dd", "14042ae82f544ce48b31e1f5739befc7", "456b7ff2159a4e82816ad2de12386fd1", "4a5d8ee6e2fa4bc79c6978afe8080cf5", "71af41b9236c45d39d47bd1f78e27efd", "5712bcedb0754c279f38d17df216949e", "fe687e506fc749eeb65eb2271c91c2d2"]} id="jXqcDAEDfZno" outputId="69283a11-3cc2-46e9-8038-517040e31645"
# 제로샷 분류(zero-shot classification) 파이프라인 생성
# 사전 정의된 레이블(label)에 대해 학습 없이 문장을 분류할 수 있음
classifier = pipeline("zero-shot-classification")

# 입력 문장을 주어진 후보 레이블 중 어떤 범주로 분류할지 예측
sequence = "This is a course about the Transformers library"
candidate_labels = ["education", "politics", "business"]

result = classifier(
    sequence     ,  # 분류할 문장
    candidate_labels=candidate_labels  # 후보 레이블 목록
)

result

# %% [markdown] id="jzChBWjG6Xms"
# ### Zero Shot Pipeline - 다국어 version

# %% colab={"base_uri": "https://localhost:8080/", "height": 322, "referenced_widgets": ["5d348d479bba42f4ba0604cfe7353dcb", "8d86584c8a4e4d3ab70b260f3794416c", "16cd131d7eaa4254a5a37a0a2ee576cb", "a99fe38906f14855b1e1f274e693a229", "00c3f6a61d1d4d36b92900b8ec63df3c", "0f9634b1038e4580929421cd7b5edb47", "d8ced0ec052d4f0a8c7d00064d643fdf", "fa3da7924b5e490ca5962909269219ac", "f9fe5c936360460c81bec92bb124bfe8", "33eff59e4c0f46c48c2b8acd4ab0bb81", "ae07b81a80f442238debe7519aa22a5f", "8a32b08b48e1420c89e0da0ce8960848", "806781356fcd4e1aa9f92411c40815d5", "dced40e4f8c643ee89ba6678f59eea73", "0674463473e24c89aaa50e7f489d67e9", "2e78ffd8e49f48edb021e1f29b9fc674", "86f3293f633742108628a60d6e035eeb", "53e8228a21ab4214b76a7dde71195787", "321e2257d13f4d4d84352a255757c971", "6c38a5184a2344d19976b135fa72bc11", "30b2bd8edaac40dd94d8adf7bb4f3f73", "4aa88b76a1294420bfa2c5724c7efbba", "83ed44ca5631450ead98db5db4828255", "70864d0425864e3aac3b84a0ca7430ba", "9551628bc9834e1682f4d5452b4638f4", "e9b47d0b23df4e23a5944ce76718cf36", "dc86afa0da9445cf80118ca5ca016bef", "f11f4986fd6345bfa18e8468559975a8", "3ad0a27b2e3c4a1089ea4578ce08f0e6", "a9b7445544d44ffeaa34a8949a1b316a", "ec15793f986f45fd921dfed8d1f28c03", "773c283869654a24b180278c35c4f41c", "394485a3111b4b7a98e2acab2f46e52c", "59b3cb3f74d746c3ae6d65612a288bdc", "60bcbb18a8ac4744bd132629fe8349ab", "7099f46191904b1ba984f0966f1ca801", "b89efcb97db04ab3a0034a1e58b007e5", "a97869d6ba8443d18a607d6773bf83ff", "d1c2cc155a34445d8ac5a36eaa068712", "22860b4c1b66460bb56295bf54ed98c7", "df2abe363b1749b4a0deec68350574a9", "029477b4c6b4450c9753913e2e29bc3a", "a17abec7767f43ec9566a55db8bec2fd", "f2b12300b88b4dba8a77b6ad60b7302e", "2c35e375772047778b31240e80a83bd6", "89faec1ce54f4fde982430eb9ee92d31", "e5f821a8d52e4fdf9676da9345f50cc9", "cd20b801b9144f838f2e079083497c77", "671b3ec6843a4d8eba5a0c79176d58e6", "03e1b95a19434688aa28a3017b86f73b", "02d1a0a64a41488e9b067185345deba3", "558e806388f34231867d37c8a591778b", "30c3ead21e6f4011a6fb5eeada28b5e9", "fd6b59171be04db2aedbd950646877fb", "76cbfa06ee264e5489d56f5f1374e1ee"]} id="pdIzdbf16Xmt" outputId="eaecb8b8-ed60-4060-db33-5fba8c4fc5bb"
classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

sequence = "2025년 대통령 선거에 누구에게 투표하겠습니까"
candidate_labels = ["건강", "문화", "정치"]
hypothesis_template = "이 텍스트는 {}에 관한 내용 입니다."

result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
result

# %% [markdown] id="Q9MuZlOpvh5Z"
# ### 개체명 인식

# %% colab={"base_uri": "https://localhost:8080/", "height": 326, "referenced_widgets": ["a641ee468a5e4d538b8b49f5eef1ad10", "3e3940879aee4e37abc8e1a14e2fd1fb", "5de7bc66c2f64b5a8ac265e10b39a24e", "de5afc62251f4a0a9caad54c61ca6c0d", "0b401dd0ba834f14a8325d0c32474989", "97222e4d20004fa9b822aebfb01cb89e", "954250c312b449d9b3611f8f776ccf80", "321000a721744b47b8160a30b7627326", "4b5f90c2f405433f8deebf9995aafbf0", "88215a2006ac49919029dc04d1963e8e", "e78aea9e2a8d4ec19de7b3cbaa220ec0", "d7aa583bc432417683eaa5d3666b0395", "fda764404c344612906d78ed894324f0", "a4801cd33b22476a9e52e99d49624ab8", "d39fe3c85e674ae280754c0d52796b51", "8a8bf74a7ee54cd19ce7cb7f238c3bd2", "31dc3b5316f34e3e9a70197f918cf3bb", "7bd3d9cd01d74a5caea3429566e5c237", "66cbc9163bd64b869bc898f84d7e97b0", "75558915b9ae4dee82c085598b7a133f", "3c3edd80b3c84abea62fbdeecf9af96a", "8296343ad5f849538691e01cea670370", "0f37ad2536f54074996694605f5ae3b2", "4e0f7ac02f0649238e041ac3eecf57e2", "f67f5dc5f9d546f687b936bbfa4ed4b3", "d065f0d7936a404ea2f66ac8fd041633", "5f8be16ebc2d445c80b1d507c5b2ec4a", "02c8cdf89f92427a984e751f47f9d194", "5bbaa721359848f9888f467aaac94b90", "bf648db4d7a04fd7aa34e99c0a0492fc", "bbecb45c5c4e497cb3496fe0fe4f1112", "172ed373186b4dbea1323b257e77f926", "1b04f6467025433db52fecd2141c6b4e", "0cfccdaf99384ae88dc6cf05cb43ec59", "598a9be6efb2497ea9cbf38d21c0eb38", "b489504f6424491fbc9fa508828eacd0", "a786c6ea5ff54c57801c2a9c8a92d1db", "5f9a4d6be8b745619057256a091ea212", "9e9e92573bbb4c27ab89f2e706f1687c", "4779a8a3280b4d43a3b3059151c6743b", "76e99190e7da403784ceed0a5c8d06fa", "8f270d6d59bd447d8a776959f7a42009", "04d1f9608ea046bf8b08f803e1e745d6", "f1f5f816551442e6a8f458d7e3b10d5a"]} id="yh-EekZEvQso" outputId="16493551-c137-4c6f-af01-6ca315d8c8f8"
# 개체명 인식(NER: Named Entity Recognition) 파이프라인 생성
# grouped_entities=True: 같은 개체로 인식된 연속된 토큰들을 하나로 묶어서 반환
ner = pipeline("ner", grouped_entities=True)

# 입력 문장에서 사람 이름, 기관명, 위치 등 고유 명사를 인식
result = ner("My name is Sylvian and I work at Hugging Face in Brooklyn")

# 결과 출력
print(result)

# %% [markdown] id="piQOP7kL6Xmu"
# ### 한글 Text 생성
# HyperCLOVAX‑SEED‑Text‑Instruct‑0.5B는 지시문 기반 텍스트-투-텍스트 모델로, 한국어 언어 및 문화 이해에 뛰어난 성능을 보입니다. 유사한 규모의 외부 경쟁 모델과 비교했을 때 수학적 성능이 향상되고, 한국어 능력이 크게 개선되었습니다. 이 모델은 HyperCLOVAX 시리즈에서 현재 출시된 모델 중 가장 작은 모델이며, 에지 디바이스와 같은 리소스가 제한된 환경에 적합한 경량 솔루션입니다. 최대 4K 토큰의 컨텍스트 길이를 지원하며, 다양한 작업에 적용 가능한 다목적 소형 모델입니다.

# %% id="S4-BCw656Xmv" colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["07d1632d5e484ea9874bb9d435c9313c", "ad63daaed1fc4f8dbb79ab10ee2b3233", "68394af3955b4041a543cc2c97fedfac", "1292619a409e44458851565a8bc4e605", "8aebd87971c649d0a60f62153d32f001", "4869870705af4dc38fa9ce873606c287", "5982ae859b8e4661a21e8f07d7ac8138", "f95be8a434d341148874bad71f09f1df", "6cfff44cd4e24d9e807d0b59e59d2c26", "d7848621cebe4a02b50c060aff2b0205", "e774bd0d5c534d2d90bfb3f8bd2f955b", "9a9f4c2b8d01449cb6f3d47058327eb5", "853b4961170e48939934c2b70a5a86f6", "ac01dcae95af42d0b489aaecb1c7ba46", "56f59a14a33c4e8598b27039b9713b25", "ff8bce8f4b8946cc9504a2c2cbb6e73e", "9e42ad742b7542389fef8eccf761bebc", "c1519d5bcfa842d2902a173c26be2b88", "1e3c3bfba4ed4fbc98c522e21335820c", "d60026cfb9a247d3acd99b31a84dd831", "6ee455f2392b4e59bba0cd90edf92946", "dcc095ae8c3a4dfea37c2861ba3c9adc", "9cdf8713ae2d472a8e8260d6454fca20", "0d33652876ef443b9effa6c162cf839b", "16a3f6ac9c2142dcbe377233274668da", "906aa7c5a5c44c78b2d99cb1152e9b7f", "b2612e86de084c1fb88c0649774164fc", "ac2ce97c0d2d4cdb921d69ea70f75754", "0a62cb84614c4af0a29dbf1eb59e7c07", "24a287dcae4e4e9885fd9f7df7697684", "98e59cf33dcf41a584ae4194ab5e4733", "52c9eb74452e4f5dbd3c85e88178eadc", "3717e125052143b18f73c3e19f646981", "156a644503d44520b573f67cc900183c", "37b5f88a554242c3bad7b14b3e55a8ee", "8db7f5c1948c4423b2f319596f5695a8", "b4c6e12ffa9b4898b19e16099a5af9e2", "3047105fd76f4b45b61b9e0b7117c07f", "48f2238c910648258e2eb08525446378", "35023e1cb7654379b73ecf20550d0bd2", "f30bf4ed116c4a9e8b6b789b9acdca81", "8f3b07fde9b7430db7e05c8bd3efc9b3", "321ad5a6ae7045daab8441a981befea3", "08c6237ed0a243f39848ec8d75cf6a93", "6f127ca5c53d4a74848bd6f7fc035661", "6834fee72ffc437da9efa3ac44ed4dd3", "815cf1dc31e346b4bbbf80b5ab3cde85", "8c3fb2fde7754696a264948209900860", "7770f38f25e74d82aac3bed3a83e2146", "6cb546408c254af3a3b6796d6f3403e2", "821ef8d2a3ba4c5c9ee23bc04a2ba26f", "9cc73541f5de4b5a87de7458be679bec", "337bc71e86ec4c509447027f001a0815", "537e217362d04a11ac7d3e0ede0717b4", "5d13bb5140494a7190ebafc5fa34275b"]} outputId="b4d45a47-0265-4eb5-b633-520b24cbdf67"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

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


# %% colab={"base_uri": "https://localhost:8080/"} id="ea0gkBdb6Xmv" outputId="9c052bca-7a43-42b8-c668-cdadb8e41ab9"
system_content = ""
user_content = "Transformer에 대해 간략히 설명해주세요."

result = generate_response(system_content, user_content)
print(result)

# %% [markdown] id="FvfXYfmKd5xY"
# ### 사전 학습된 모델을 이용한 챗봇
#

# %% colab={"base_uri": "https://localhost:8080/"} id="O-jDWRQaeEmt" outputId="72f4ceaf-3ba0-43f0-eab8-70f753524f21"
import re

# 챗봇과 대화하는 함수 정의 (간단한 버전)
def chat_with_bot():
    system_content = ""  # 시스템 메시지 (필요시 설정)

    while True:
        # 사용자로부터 입력 받기
        user_content = input("You: ")

        # 종료 조건: 사용자가 quit, exit, bye 입력 시 대화 종료
        if user_content.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        try:
            # generate_response 함수를 사용하여 응답 생성
            result = generate_response(system_content, user_content)

            # regex로 "assistant" 이후의 내용만 추출하고 정리
            cleaned_result = re.sub(r'.*?assistant\s*', '', result, flags=re.IGNORECASE)
            cleaned_result = re.sub(r'^(tool_list|system|user|role|content)\s*\n?', '', cleaned_result, flags=re.IGNORECASE | re.MULTILINE)
            # 사용자 입력이 그대로 출력된 부분 제거
            cleaned_result = re.sub(rf'^{re.escape(user_content)}\s*\n?', '', cleaned_result, flags=re.MULTILINE)
            cleaned_result = re.sub(r'\n+', '\n', cleaned_result).strip()

            print("Chatbot:", cleaned_result)
            print()

        except Exception as e:
            print(f"Chatbot: 죄송합니다. 오류가 발생했습니다: {e}")

# 대화 시작
chat_with_bot()

# %% id="ommAYenz3lB5"
