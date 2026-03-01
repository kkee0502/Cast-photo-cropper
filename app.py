import streamlit as st
import numpy as np
import cv2
from PIL import Image
from rembg import remove
import io

# 페이지 레이아웃 설정
st.set_page_config(page_title="Dental Model Trimmer", layout="centered")

def process_dental_image(input_image):
    # 1. 배경 제거 및 투명화 (rembg 사용)
    output = remove(input_image)
    
    # 2. 투명한 배경을 흰색으로 채우기
    white_bg = Image.new("RGBA", output.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, output).convert("RGB")
    
    # 3. OpenCV를 이용한 자동 크롭 (모델 부분만 남기기)
    img_np = np.array(composite)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 배경이 아닌 부분(250보다 어두운 부분)을 검출
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 모든 윤곽선을 포함하는 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # 여백(padding) 추가
        p = 20
        x, y = max(0, x-p), max(0, y-p)
        w, h = min(img_np.shape[1]-x, w+p*2), min(img_np.shape[0]-y, h+p*2)
        
        # 크롭 적용
        img_np = img_np[y:y+h, x:x+w]
        
    return Image.fromarray(img_np)

st.title("🦷 치아 모델 자동 편집기")
st.write("스캔 화면에서 치아 모델만 추출하여 배경을 흰색으로 변경합니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="원본 이미지", use_container_width=True)
    
    if st.button("편집 시작"):
        with st.spinner("AI 처리 중..."):
            result = process_dental_image(raw_image)
            st.image(result, caption="편집 완료", use_container_width=True)
            
            # 파일 다운로드 준비
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button(
                label="이미지 저장하기",
                data=buf.getvalue(),
                file_name="processed_model.png",
                mime="image/png"
            )

# 하단 구분용 기호
st.write("-")
st.write("-")
