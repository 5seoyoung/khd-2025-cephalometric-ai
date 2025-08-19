# -*- coding: utf-8 -*-
"""
Cephalometric AI - Streamlit Demo UI
측면두부규격방사선사진 AI 분석 데모 인터페이스

사용법: streamlit run src/demo/streamlit_app.py
"""

import streamlit as st
import os
import sys
import time
from PIL import Image, ImageDraw
import json
import base64
from io import BytesIO

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

try:
    from src.core.integration_pipeline import CephalometricPipeline
except ImportError:
    st.error("모듈을 찾을 수 없습니다. 프로젝트 구조를 확인해주세요.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Cephalometric AI Demo",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (녹색 포인트 + 흰색 배경)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F0FFF0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .status-normal { color: #2E8B57; font-weight: bold; }
    .status-high { color: #FF6347; font-weight: bold; }
    .status-low { color: #4169E1; font-weight: bold; }
    .classification-result {
        background: #F8F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        text-align: center;
    }
    .performance-badge {
        background: #2E8B57;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """세션 상태 초기화"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = True

def create_landmark_overlay(image, landmarks, highlight_points=None):
    """이미지에 랜드마크를 오버레이합니다."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for name, (x, y) in landmarks.items():
        # 하이라이트 포인트는 다른 색상
        if highlight_points and name in highlight_points:
            color = 'blue'
            radius = 6
        else:
            color = 'red'
            radius = 4
            
        # 점 그리기
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=color, outline='white', width=2)
        
        # 라벨 그리기 (작은 폰트)
        draw.text((x+8, y-8), name, fill=color)
    
    return img_copy

def display_clinical_metrics(metrics):
    """임상 지표를 카드 형태로 표시"""
    st.markdown("### 📊 임상 지표")
    
    cols = st.columns(2)
    
    for i, (metric_name, data) in enumerate(metrics.items()):
        col = cols[i % 2]
        
        with col:
            # 상태에 따른 아이콘과 색상
            status = data["status"]
            if status == "normal":
                icon = "✅"
                status_class = "status-normal"
            elif status == "high":
                icon = "⬆️"
                status_class = "status-high"
            else:  # low
                icon = "⬇️"
                status_class = "status-low"
            
            # 메트릭 카드
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{icon} {metric_name}</h4>
                    <p><span class="{status_class}">{data['value']:.1f}°</span></p>
                    <p><small>정상 범위: {data['normal_range'][0]}-{data['normal_range'][1]}°</small></p>
                    <p><small>{data.get('clinical_significance', '')}</small></p>
                </div>
            """, unsafe_allow_html=True)

def display_classification_result(classification):
    """분류 결과를 표시"""
    st.markdown("### 🎯 분류 결과")
    
    label = classification["predicted_label"]
    confidence = classification["confidence"]
    anb_value = classification["anb_value"]
    
    # 분류별 색상 및 설명
    class_info = {
        "Class I": {"color": "#2E8B57", "desc": "골격적으로 정상"},
        "Class II": {"color": "#FF6347", "desc": "골격적으로 상악 과성장"},
        "Class III": {"color": "#4169E1", "desc": "골격적으로 하악 과성장"}
    }
    
    color = class_info.get(label, {}).get("color", "#2E8B57")
    description = class_info.get(label, {}).get("desc", "")
    
    st.markdown(f"""
        <div class="classification-result">
            <h2 style="color: {color};">{label}</h2>
            <h4>{description}</h4>
            <p><strong>신뢰도:</strong> {confidence*100:.1f}%</p>
            <p><strong>ANB 각도:</strong> {anb_value:.1f}°</p>
            <p><small>{classification.get('classification_basis', '')}</small></p>
        </div>
    """, unsafe_allow_html=True)
    
    # 확률 분포 표시
    st.markdown("#### 분류 확률")
    probs = classification["probabilities"]
    
    for class_name, prob in probs.items():
        progress_color = color if class_name == label else "#E0E0E0"
        st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")

def load_demo_image():
    """대표 도면 이미지를 로드합니다."""
    demo_path = "data/sample_images/demo_xray.jpg"
    if os.path.exists(demo_path):
        return Image.open(demo_path)
    else:
        # 대체 이미지 (회색 사각형)
        img = Image.new('RGB', (800, 600), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((350, 280), "Demo Image\nPlaceholder", 
                 fill='darkgray', anchor='mm')
        return img

def main():
    """메인 UI 함수"""
    initialize_session_state()
    
    # 헤더
    st.markdown("""
        <div class="main-header">
            <h1>🦷 Cephalometric AI - Demo</h1>
            <p>측면두부규격방사선사진 자동 분석 시스템</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # 데모 모드 토글
        demo_mode = st.toggle("데모 모드", value=True, 
                             help="오프라인 시뮬레이션 모드")
        st.session_state.demo_mode = demo_mode
        
        if not demo_mode:
            st.warning("연구 모드는 안심존에서만 사용 가능합니다")
            demo_mode = True
        
        # 파이프라인 초기화
        if st.session_state.pipeline is None:
            with st.spinner("파이프라인 초기화 중..."):
                try:
                    st.session_state.pipeline = CephalometricPipeline(
                        demo_mode=demo_mode, 
                        seed=42
                    )
                    st.success("✅ 초기화 완료")
                except Exception as e:
                    st.error(f"❌ 초기화 실패: {e}")
                    st.stop()
        
        st.markdown("---")
        
        # 환자 정보 입력
        st.markdown("### 👤 환자 정보")
        patient_age = st.number_input("나이", min_value=1, max_value=99, value=25)
        patient_sex = st.selectbox("성별", ["F", "M", "U"], index=0)
        patient_id = st.text_input("환자 ID", value="DEMO001")
        
        meta = {
            "age": patient_age,
            "sex": patient_sex,
            "patient_id": patient_id
        }
        
        st.markdown("---")
        
        # 앵커 포인트 설정
        st.markdown("### 🔧 고급 설정")
        use_anchors = st.checkbox("FH 기준선 수동 보정", 
                                 help="Or, Po 두 점을 수동으로 지정하여 Frankfort Horizontal plane 보정")
        
        anchors = None
        if use_anchors:
            st.info("Or(Orbitale), Po(Porion) 좌표를 입력하세요")
            or_x = st.number_input("Or X", value=400, min_value=0, max_value=2000)
            or_y = st.number_input("Or Y", value=200, min_value=0, max_value=2000)
            po_x = st.number_input("Po X", value=300, min_value=0, max_value=2000)
            po_y = st.number_input("Po Y", value=210, min_value=0, max_value=2000)
            
            anchors = {
                "Or": (float(or_x), float(or_y)),
                "Po": (float(po_x), float(po_y))
            }
    
    # 메인 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 이미지 입력")
        
        # 이미지 선택 방식
        input_method = st.radio(
            "입력 방식 선택:",
            ["대표 도면 사용", "파일 업로드"],
            horizontal=True
        )
        
        selected_image = None
        
        if input_method == "대표 도면 사용":
            if st.button("대표 도면 로드", type="primary"):
                selected_image = load_demo_image()
                st.session_state.input_image = selected_image
                
        else:  # 파일 업로드
            uploaded_file = st.file_uploader(
                "X-ray 이미지 업로드",
                type=["jpg", "jpeg", "png"],
                help="측면두부규격방사선사진을 업로드하세요"
            )
            
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                st.session_state.input_image = selected_image
        
        # 이미지 표시
        if hasattr(st.session_state, 'input_image'):
            st.image(st.session_state.input_image, 
                    caption="입력 이미지", 
                    use_container_width=True)
            
            # 분석 실행 버튼
            if st.button("🚀 분석 시작", type="primary", use_container_width=True):
                with st.spinner("분석 중..."):
                    try:
                        start_time = time.time()
                        
                        # 파이프라인 실행
                        result = st.session_state.pipeline.run(
                            st.session_state.input_image,
                            meta=meta,
                            anchors=anchors
                        )
                        
                        execution_time = time.time() - start_time
                        
                        if "error" in result:
                            st.error(f"❌ 분석 실패: {result['error']['message']}")
                        else:
                            st.session_state.analysis_results = result
                            
                            # 성능 배지 표시
                            total_time = result["performance"]["total_time_ms"]
                            st.markdown(f"""
                                <div class="performance-badge">
                                    ⚡ {total_time:.1f}ms 완료
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("✅ 분석 완료!")
                            
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류 발생: {e}")
    
    with col2:
        st.markdown("### 📋 분석 결과")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # 탭으로 결과 구분
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 분류", "📊 지표", "📍 랜드마크", "⚡ 성능"])
            
            with tab1:
                display_classification_result(results["classification"])
            
            with tab2:
                display_clinical_metrics(results["clinical_metrics"])
            
            with tab3:
                st.markdown("#### 랜드마크 위치")
                
                # 랜드마크 오버레이 이미지 생성
                if hasattr(st.session_state, 'input_image'):
                    landmarks = results["landmarks"]["coordinates"]
                    highlight = ["Or", "Po"] if anchors else None
                    
                    overlay_img = create_landmark_overlay(
                        st.session_state.input_image, 
                        landmarks,
                        highlight_points=highlight
                    )
                    
                    st.image(overlay_img, 
                            caption=f"랜드마크 표시 ({len(landmarks)}개 점)", 
                            use_container_width=True)
                
                # 좌표 테이블
                with st.expander("좌표 상세 정보"):
                    landmarks = results["landmarks"]["coordinates"]
                    coords_data = []
                    for name, (x, y) in landmarks.items():
                        coords_data.append({
                            "랜드마크": name,
                            "X": f"{x:.1f}",
                            "Y": f"{y:.1f}"
                        })
                    st.dataframe(coords_data, use_container_width=True)
            
            with tab4:
                st.markdown("#### 성능 분석")
                
                perf = results["performance"]
                quality = results["quality"]
                
                # 성능 메트릭
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("총 처리 시간", f"{perf['total_time_ms']:.1f}ms")
                    st.metric("추론 시간", f"{perf['inference_ms']:.1f}ms")
                
                with col_b:
                    st.metric("품질 점수", f"{quality['overall_score']:.3f}")
                    st.metric("추론 모드", results["landmarks"]["inference_mode"])
                
                # 품질 평가
                st.markdown("#### 품질 평가")
                st.info(quality["recommendation"])
                
                if quality["warnings"]:
                    for warning in quality["warnings"]:
                        st.warning(f"⚠️ {warning}")
        
        else:
            st.info("👆 이미지를 선택하고 분석을 시작해주세요")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🦷 Cephalometric AI Demo | 
            <a href="#" style="color: #2E8B57;">GitHub</a> | 
            <a href="#" style="color: #2E8B57;">Documentation</a></p>
            <p><small>이 데모는 교육 및 연구 목적으로 제작되었습니다.</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()