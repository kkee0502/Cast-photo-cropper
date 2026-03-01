import streamlit as st
from PIL import Image
import requests
import io
import base64

# 페이지 설정
st.set_page_config(page_title="AI Dental Editor", layout="wide")

# Hugging Face API 설정 (ControlNet 기반 모델 예시)
# 실제 사용 시 본인의 API 키를 입력하거나 Streamlit Secrets에 저장하세요.
API_URL = "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny"

def query(payload, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# UI 구성
st.title("🦷 치아 교정 전/후 시뮬레이션 편집기")
st.markdown("---")

# 사이드바 설정
st.sidebar.header("⚙️ API 설정")
hf_token = st.sidebar.text_input("Hugging Face Token을 입력하세요", type="password")
st.sidebar.info("발급처: huggingface.co/settings/tokens")

# 메인 화면 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.header("1. 교정 전 (Before)")
    before_file = st.file_uploader("원본 이미지를 업로드하세요", type=['jpg', 'png'], key="before")
    if before_file:
        st.image(before_file, caption="Original Scan", use_container_width=True)

with col2:
    st.header("2. 스타일 참조 (After)")
    after_file = st.file_uploader("목표 스타일 이미지를 업로드하세요", type=['jpg', 'png'], key="after")
    if after_file:
        st.image(after_file, caption="Target Style", use_container_width=True)

st.markdown("---")

# 실행 버튼
if st.button("✨ AI 편집 시작"):
    if before_file and after_file and hf_token:
        with st.spinner("AI가 치열을 분석하고 이미지를 생성 중입니다..."):
            try:
                # 이미지 읽기 및 인코딩
                before_bytes = before_file.getvalue()
                # 간단한 예시를 위한 프롬프트 구성
                payload = {
                    "inputs": "perfectly aligned teeth, dental clinic result, high quality, medical scan",
                    "image": base64.b64encode(before_bytes).decode("utf-8")
                }
                
                # API 호출
                result_bytes = query(payload, hf_token)
                result_img = Image.open(io.BytesIO(result_bytes))
                
                st.success("편집 완료!")
                st.image(result_img, caption="AI Generated Result", use_container_width=True)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("전/후 사진과 API 토큰이 모두 필요합니다.")

st.markdown("---")
st.caption("© 2026 AI Dental Simulation Tool -")

import joblib

# 모델 저장 (예: CatBoost 모델)
joblib.dump(model, 'credit_model.pkl')
# 스케일러나 인코더가 있다면 그것도 저장해야 함 (예: scaler.pkl)
