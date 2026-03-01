import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- 설정 ---
st.set_page_config(page_title="Dental Model Trimmer", layout="wide")

def process_dental_model(image_np):
    # 1. 원본 크기 저장
    h, w = image_np.shape[:2]
    
    # 2. 배경 제거 (GrabCut 또는 Threshold 사용)
    # 치아 모델이 밝고 배경이 어두운 경우를 가정하여 단순화된 마스크 생성
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 가장 큰 컨투어(치아 모델 본체) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 4. 둥근 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    # 하단을 둥글게 깎기 위해 타원형이나 둥근 사각형 그리기
    # 여기서는 이미지 하단 20% 지점부터 둥글게 처리하는 로직
    mask = cv2.ellipse(mask, (w//2, h-h//4), (w//2, h//2), 0, 0, 360, 255, -1)
    # 기존 모델 영역과 둥근 마스크의 교집합
    final_mask = cv2.bitwise_and(thresh, mask)
    
    # 5. 배경을 흰색으로 채우기
    result = np.full_like(image_np, 255) # 흰색 배경
    # 마스크 영역만 원본에서 복사
    idx = (final_mask > 0)
    result[idx] = image_np[idx]
    
    return result

# --- UI ---
st.title("🦷 치아 모델 자동 트리머 (Background & Rounding)")
st.write("치아 스캔 사진을 업로드하면 배경을 제거하고 하단을 둥글게 다듬어줍니다.")

uploaded_file = st.file_uploader("이미지 업로드 (.jpg, .png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 (Before)")
        st.image(img, use_container_width=True)
        
    with col2:
        st.subheader("결과 (After)")
        if st.button("자동 편집 실행"):
            with st.spinner("이미지 처리 중..."):
                processed_img_np = process_dental_model(img_np)
                processed_img = Image.fromarray(processed_img_np)
                st.image(processed_img, use_container_width=True)
                
                # 다운로드 버튼
                buf = io.BytesIO()
                processed_img.save(buf, format="PNG")
                st.download_button("결과 이미지 다운로드", buf.getvalue(), "processed_model.png", "image/png")

# 하단 공백
for _ in range(5):
    st.write("-")
