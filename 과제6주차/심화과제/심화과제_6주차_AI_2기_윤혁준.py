import base64
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# BLIP 모델 및 프로세서 초기화
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 모델 초기화 (gpt-4-mini 모델 사용)
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Streamlit UI
st.title("소비 패턴기반 패션 추천 서비스")
st.write("내가 구매한 아이템 사진을 업로드해주세요")

# 세션 상태 초기화 (이미지가 업로드되지 않았다면)
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# 이미지 업로드 (여러 이미지 업로드 가능)
uploaded_images = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], key="image_uploader", accept_multiple_files=True)

# 새로운 이미지를 업로드했을 때 세션 상태를 초기화
if uploaded_images:
    # 업로드된 이미지들을 세션 상태에 저장
    st.session_state.uploaded_images = uploaded_images

# 이미지가 업로드된 경우, 이미지 표시
if st.session_state.uploaded_images:
    captions = []  # 이미지를 보고 생성된 캡션을 저장할 리스트

    # 한 줄에 3개의 이미지를 배치할 수 있도록 st.columns 사용
    num_cols = 3  # 한 줄에 3개 이미지
    cols = st.columns(num_cols)

    # 업로드된 이미지를 표시하고, 캡션 생성
    for idx, img_file in enumerate(st.session_state.uploaded_images):
        img = Image.open(img_file).convert("RGB")
        # BLIP 모델을 사용하여 이미지 캡션 생성
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
        with cols[idx % num_cols]:  # 순차적으로 컬럼에 배치
            st.image(img, caption=f"{caption}", use_container_width=True)

    # 추천 항목 체크박스
    st.write("추천 받고 싶은 카테고리를 고르세요")
    recommended_items = []
    
    if st.checkbox("하의"):
        recommended_items.append("하의")
    if st.checkbox("신발"):
        recommended_items.append("신발")
    if st.checkbox("상의"):
        recommended_items.append("상의")

    # 결합된 프롬프트 만들기
    if recommended_items:

        # 모든 이미지 캡션을 하나로 통합
        all_captions = " ".join(captions)

        # GPT 요청을 위한 messages 형식으로 데이터 준비
        prompt = f"사용자가 구매한 아이템을 보고 {', '.join(recommended_items)} 패션 아이템을 추천해주세요. 사용자가 구매한 목록들: {all_captions}\n"

        # 사용자 메시지를 HumanMessage 객체로 생성
        message = HumanMessage(content=prompt)

        # 모델 호출 (로딩 스피너 적용)
        with st.spinner("아이템을 고르고 있는 중 입니다..."):
                result = chat_model.invoke([message])

        # GPT 응답 출력 (챗봇 형식으로)
        recommendation = result.content.strip()
        st.chat_message("assistant").markdown(f"**Assistant**: {recommendation}")
    else:
        st.write("")
else:
    st.write("")