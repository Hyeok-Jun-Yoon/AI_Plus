import base64
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

st.title("Images based question Chatbot")

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# 상태 초기화
# 업로드된 이미지를 저장하기 위한 리스트
if "images" not in st.session_state:
    st.session_state["images"] = []
# 대화 기록(질문과 답변)을 저장하기 위한 리스트
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 여러 이미지 업로드
uploaded_images = st.file_uploader(
    "사진들을 올려주세요! (여러 장 선택 가능)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# 이미지 상태 저장
if uploaded_images:
    st.session_state["images"] = [
        {
            "name": image.name,
            "data": base64.b64encode(image.read()).decode("utf-8")
        }
        for image in uploaded_images
    ]
    st.success(f"{len(uploaded_images)}개의 이미지가 성공적으로 업로드되었습니다.")

# 업로드된 이미지 출력
if st.session_state["images"]:
    for img in st.session_state["images"]:
        st.image(base64.b64decode(img["data"]))

# 기존 대화 출력 (실시간 채팅처럼 표시)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# 사용자 질문 입력
if user_input := st.chat_input("질문을 입력하세요:"):
    # 사용자 질문 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # GPT 요청 생성
    messages = [
        {"type": "text", "text": user_input},
        *[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}
            }
            for img in st.session_state["images"]
        ]
    ]

    # 모델 호출
    with st.spinner("GPT가 응답을 생성 중입니다..."):
        message = HumanMessage(content=messages)
        result = model.invoke([message])

    # GPT 응답 추가
    st.session_state["messages"].append({"role": "assistant", "content": result.content})
    with st.chat_message("assistant"):
        st.markdown(result.content)
