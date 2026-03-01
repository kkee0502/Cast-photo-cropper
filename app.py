import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- 설정 ---
st.set_page_config(page_title="AI Dental Model Editor", layout="wide")

# --- 이미지 처리 함수 ---
def process_image(image, background_color=(255, 255, 255), corner_radius=20):
    # 1. 배경 제거 (GrabCut 알고리즘 활용)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10) # 초기 사각형 설정
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    image_fg = image*mask2[:,:,np.newaxis]

    # 2. 배경을 하얗게 변경
    background = np.full(image.shape, background_color, dtype=np.uint8)
    image_bg_removed = cv2.bitwise_or(image_fg, background, mask=cv2.bitwise_not(mask2))

    # 3. 모서리를 둥글게 처리
    # (모서리를 둥글게 처리하는 로직은 다소 복잡하여, 여기서는 간단하게 모서리를 자르는 방식으로 시뮬레이션합니다.)
    # 실제 구현을 위해서는 마스크를 활용하여 모서리를 부드럽게 깎아내는 과정이 필요합니다.
    # ... (모서리 둥글게 처리 로직 추가 예정) ...

    return image_bg_removed

# --- UI 레이아웃 ---
st.title("🦷 AI 치아 모델 편집기")
st.write("전 사진을 업로드하여 배경을 하얗게 바꾸고 치아 모델 주변을 둥글게 편집해보세요.")
st.markdown("---")

# 레이아웃 분할
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📸 전 사진 (Before)")
    uploaded_file = st.file_uploader("편집할 사진을 선택하세요", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="전 이미지", use_container_width=True)

with right_col:
    st.subheader("🪄 편집 결과 (After)")
    
    # 세부 조절 슬라이더 (옵션)
    # corner_radius = st.slider("모서리 둥글기", 0, 50, 20)
    
    if st.button("이미지 편집하기"):
        if uploaded_file:
            with st.spinner("AI가 이미지를 분석하고 편집 중입니다... 잠시만 기다려주세요."):
                # 이미지 처리 함수 호출
                input_image_np = np.array(input_image)
                processed_image_np = process_image(input_image_np)
                processed_image = Image.fromarray(processed_image_np)
                
                st.success("편집이 완료되었습니다!")
                st.image(processed_image, caption="편집된 결과 (After)", use_container_width=True)
        else:
            st.error("사진을 업로드해주세요.")

# 하단 구분선 및 여백
st.markdown("---")
st.write("-")
st.write("-")
st.write("-")
