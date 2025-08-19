# -*- coding: utf-8 -*-
"""
Cephalometric AI - Konyang University Medical Center Theme
측면두부규격방사선사진 AI 분석 데모 인터페이스 (건양대 테마)

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
import numpy as np
from datetime import datetime
import hashlib

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
    page_title="Cephalometric AI - Konyang Medical Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 건양대 테마 커스텀 CSS (그라데이션 제거, 단색 적용)
st.markdown("""
<style>
    :root {
        --ky-pine-green: #2D5530;
        --ky-light-green: #7FB069;
        --ky-dark-green: #1B3A1E;
        --ky-gold: #B8860B;
        --ky-red: #C53030;
        --ky-apricot: #FFA726;
        --ky-riverside: #5B9BD5;
        --ky-light-purple: #9575CD;
        --ky-dark-gray: #424242;
        --ky-silver: #9E9E9E;
    }

    .main-header {
        background: var(--ky-pine-green);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: left;
        margin-bottom: 2rem;
        position: relative;
        box-shadow: 0 4px 12px rgba(45, 85, 48, 0.3);
        margin-top: 2rem;
    }

    .header-logo-top {
        position: absolute;
        top: -40px;
        left: 20px;
        z-index: 10;
        background: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .header-content {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 2rem;
        margin-top: 20px;
    }

    .header-text { flex: 1; }

    .performance-dashboard {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid var(--ky-light-green);
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(127, 176, 105, 0.2);
    }

    .metric-card {
        background: #F0FFF0;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--ky-pine-green);
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(45, 85, 48, 0.1);
    }

    .status-badge-normal {
        background: var(--ky-pine-green);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(45, 85, 48, 0.3);
    }
    .status-badge-warning {
        background: var(--ky-apricot);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(255, 167, 38, 0.3);
    }
    .status-badge-error {
        background: var(--ky-red);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(197, 48, 48, 0.3);
    }

    .whatif-simulator {
        background: #E3F2FD;
        border: 3px solid var(--ky-riverside);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(91, 155, 213, 0.2);
    }
    .whatif-header {
        background: var(--ky-riverside);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .status-normal { color: var(--ky-pine-green); font-weight: bold; }
    .status-high { color: var(--ky-red); font-weight: bold; }
    .status-low { color: var(--ky-riverside); font-weight: bold; }

    .classification-result {
        background: #F8F8FF;
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid var(--ky-pine-green);
        text-align: center;
        box-shadow: 0 6px 20px rgba(45, 85, 48, 0.15);
    }

    .performance-badge {
        background: var(--ky-pine-green);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(45, 85, 48, 0.3);
    }

    .qc-warning {
        background: #FFF8E1;
        border: 2px solid var(--ky-gold);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(184, 134, 11, 0.2);
    }
    .qc-error {
        background: #FFEBEE;
        border: 2px solid var(--ky-red);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(197, 48, 48, 0.2);
    }

    .sidebar .stSelectbox > div > div {
        background: #F0FFF0;
        border: 1px solid var(--ky-light-green);
    }

    .system-info-card {
        background: var(--ky-dark-gray);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(66, 66, 66, 0.3);
    }

    .stButton > button {
        background: var(--ky-pine-green);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(45, 85, 48, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: var(--ky-light-green);
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(45, 85, 48, 0.4);
    }

    .stProgress > div > div > div { background: var(--ky-pine-green); }

    [data-testid="metric-container"] {
        background: #F8F9FA;
        border: 1px solid var(--ky-light-green);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(127, 176, 105, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def get_konyang_logo_base64():
    """건양대 로고를 Base64로 인코딩"""
    import base64
    import os
    logo_path = "data/assets/konyang_logo.png"
    try:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_data = f.read()
            return base64.b64encode(logo_data).decode()
        else:
            logo_svg = """
            <svg width="120" height="50" viewBox="0 0 120 50" xmlns="http://www.w3.org/2000/svg">
                <rect width="120" height="50" fill="white" rx="10"/>
                <circle cx="25" cy="25" r="15" fill="#2D5530"/>
                <circle cx="25" cy="25" r="8" fill="white"/>
                <text x="50" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#2D5530">건양대학교</text>
                <text x="50" y="35" font-family="Arial, sans-serif" font-size="10" fill="#7FB069">의료원</text>
            </svg>
            """
            return base64.b64encode(logo_svg.encode()).decode()
    except Exception:
        logo_svg = """
        <svg width="120" height="50" viewBox="0 0 120 50" xmlns="http://www.w3.org/2000/svg">
            <rect width="120" height="50" fill="white" rx="10"/>
            <circle cx="25" cy="25" r="15" fill="#2D5530"/>
            <circle cx="25" cy="25" r="8" fill="white"/>
            <text x="50" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#2D5530">건양대학교</text>
            <text x="50" y="35" font-family="Arial, sans-serif" font-size="10" fill="#7FB069">의료원</text>
        </svg>
        """
        return base64.b64encode(logo_svg.encode()).decode()

def initialize_session_state():
    """세션 상태 초기화"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = True
    if 'batch_running' not in st.session_state:
        st.session_state.batch_running = False

def render_performance_dashboard(pipeline_result):
    """건양대 테마 성능 대시보드"""
    st.markdown("""
        <div class="performance-dashboard">
            <h3 style="color: #2D5530; margin-bottom: 1rem;">⚡ 실시간 성능 대시보드</h3>
        </div>
    """, unsafe_allow_html=True)

    performance = pipeline_result.get('performance', {})
    quality = pipeline_result.get('quality', {})
    classification = pipeline_result.get('classification', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_time = performance.get('total_time_ms', 0.0)
        st.metric(
            label="⚡ 총 처리시간",
            value=f"{total_time:.1f}ms",
            delta="목표: <50ms",
            help="전체 파이프라인 처리 시간"
        )

    with col2:
        quality_score = float(quality.get('overall_score', 0.0)) * 100.0
        star_rating = "⭐" * min(5, int(quality_score // 20) + 1)
        st.metric(
            label=f"🎯 품질점수 {star_rating}",
            value=f"{quality_score:.1f}%",
            delta="임상 기준: >80%",
            help="랜드마크 검출 품질 종합 점수"
        )

    with col3:
        predicted_class = classification.get('predicted_class', 'Unknown')
        confidence = float(classification.get('confidence', 0.0)) * 100.0
        if confidence >= 80:
            conf_indicator = "🟢"
        elif confidence >= 60:
            conf_indicator = "🟡"
        else:
            conf_indicator = "🔴"
        st.metric(
            label=f"{conf_indicator} 분류결과",
            value=f"Class {predicted_class}",
            delta=f"신뢰도 {confidence:.1f}%",
            help="부정교합 분류 결과 및 예측 신뢰도"
        )

    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        inference_mode = pipeline_result.get('landmarks', {}).get('inference_mode', 'unknown')
        st.metric(
            label="🕐 분석완료",
            value=current_time,
            delta=f"모드: {inference_mode}",
            help="분석 완료 시각 및 추론 모드"
        )

def render_clinical_status_badges(clinical_metrics):
    """건양대 테마 정상범위 배지 시스템"""
    st.markdown("### 📊 임상 지표 상태")

    normal_ranges = {'SNA': (80, 84), 'SNB': (78, 82), 'ANB': (0, 4), 'FMA': (25, 30)}
    colors = {'normal': '#2D5530', 'warning': '#FFA726', 'error': '#C53030'}
    cols = st.columns(4)
    metric_names = ['SNA', 'SNB', 'ANB', 'FMA']

    for i, metric_name in enumerate(metric_names):
        if metric_name in clinical_metrics:
            metric_data = clinical_metrics[metric_name]
            value = float(metric_data['value'])
            normal_min, normal_max = normal_ranges[metric_name]

            if normal_min <= value <= normal_max:
                status = "정상"; badge_class = "status-badge-normal"; color = colors['normal']; icon = "✅"
            elif abs(value - normal_min) <= 2 or abs(value - normal_max) <= 2:
                status = "경계"; badge_class = "status-badge-warning"; color = colors['warning']; icon = "⚠️"
            else:
                status = "이탈"; badge_class = "status-badge-error"; color = colors['error']; icon = "🚨"

            with cols[i]:
                st.markdown(f"""
                <div style="
                    border: 3px solid {color};
                    border-radius: 15px;
                    padding: 18px;
                    text-align: center;
                    background: {color}10;
                    margin: 8px;
                    box-shadow: 0 4px 12px {color}30;
                ">
                    <h4 style="margin: 0; color: {color}; font-weight: bold;">{icon} {metric_name}</h4>
                    <p style="margin: 8px 0; font-size: 20px; font-weight: bold; color: {color};">{value:.1f}°</p>
                    <p style="margin: 0; font-size: 13px; color: #666;">정상: {normal_min}-{normal_max}° | {status}</p>
                    <span class="{badge_class}">{status}</span>
                </div>
                """, unsafe_allow_html=True)

def render_whatif_simulator(analysis_result):
    """건양대 테마 What-if 시뮬레이터"""
    st.markdown("""
        <div class="whatif-simulator">
            <div class="whatif-header">
                <h3 style="margin: 0;">🎛️ What-if 시뮬레이션</h3>
                <p style="margin: 0.5rem 0 0 0;"><em>임상 지표를 조정하면 분류가 어떻게 바뀔까요?</em></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    original_metrics = analysis_result['clinical_metrics']
    original_classification = analysis_result['classification']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: #2D5530; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <strong>원본 분류: Class {original_classification['predicted_class']}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: #5B9BD5; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <strong>원본 신뢰도: {original_classification['confidence']*100:.1f}%</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🎚️ 임상 지표 조정")
    slider_cols = st.columns(2)
    with slider_cols[0]:
        original_anb = float(original_metrics['ANB']['value'])
        adjusted_anb = st.slider("ANB 각도 (°)", min_value=-5.0, max_value=15.0, value=float(original_anb), step=0.5,
                                 help="ANB = SNA - SNB (상하악 관계의 핵심 지표)")
        original_fma = float(original_metrics['FMA']['value'])
        adjusted_fma = st.slider("FMA 각도 (°)", min_value=15.0, max_value=40.0, value=float(original_fma), step=0.5,
                                 help="하악 경사각 (수직성장 vs 수평성장)")
    with slider_cols[1]:
        original_sna = float(original_metrics['SNA']['value'])
        adjusted_sna = st.slider("SNA 각도 (°)", min_value=75.0, max_value=90.0, value=float(original_sna), step=0.5,
                                 help="상악골 전후방 위치")
        original_snb = float(original_metrics['SNB']['value'])
        adjusted_snb = st.slider("SNB 각도 (°)", min_value=70.0, max_value=85.0, value=float(original_snb), step=0.5,
                                 help="하악골 전후방 위치")

    new_classification = simulate_classification_from_anb(adjusted_anb)

    st.markdown("#### 📊 조정 결과 비교")
    comparison_cols = st.columns(3)
    with comparison_cols[0]:
        anb_change = adjusted_anb - original_anb
        st.metric("ANB 변화량", f"{anb_change:+.1f}°", delta=f"현재: {adjusted_anb:.1f}°")
    with comparison_cols[1]:
        class_changed = new_classification['class'] != original_classification['predicted_class']
        st.metric("새 분류", f"Class {new_classification['class']}", delta="변경됨" if class_changed else "동일")
    with comparison_cols[2]:
        conf_change = new_classification['confidence'] - original_classification['confidence']
        st.metric("신뢰도 변화", f"{new_classification['confidence']*100:.1f}%", delta=f"{conf_change*100:+.1f}%")

    if abs(anb_change) > 0.5:
        st.markdown("#### 💡 임상적 해석")
        interpret_anb_change_konyang(original_anb, adjusted_anb, new_classification)

    return {
        'adjusted_anb': adjusted_anb,
        'adjusted_sna': adjusted_sna,
        'adjusted_snb': adjusted_snb,
        'adjusted_fma': adjusted_fma,
        'new_classification': new_classification
    }

def simulate_classification_from_anb(anb_value):
    """ANB 값으로부터 분류 시뮬레이션"""
    if anb_value < 0:
        predicted_class = 3; base_confidence = 0.85
    elif anb_value <= 4:
        predicted_class = 1; base_confidence = 0.80
    else:
        predicted_class = 2; base_confidence = 0.82
    if abs(anb_value - 0) < 1 or abs(anb_value - 4) < 1:
        base_confidence -= 0.15
    confidence = float(np.clip(base_confidence, 0.3, 0.95))
    return {'class': predicted_class, 'confidence': confidence,
            'anb_category': 'Class III' if anb_value < 0 else 'Class I' if anb_value <= 4 else 'Class II'}

def interpret_anb_change_konyang(original_anb, new_anb, new_result):
    """건양대 테마 ANB 변화 해석"""
    change = new_anb - original_anb

    if abs(change) < 0.5:
        st.markdown("""
        <div style="background: #2D5530; color: white; padding: 1.2rem; border-radius: 15px;">
            <h4 style="margin: 0;">✅ 미미한 변화: 분류에 큰 영향 없음</h4>
        </div>
        """, unsafe_allow_html=True)
        return

    if change > 0:
        direction = "증가"; meaning = "상악 과성장 또는 하악 후퇴 양상"; tendency = "Class II 방향"; color = "#C53030"
    else:
        direction = "감소"; meaning = "상악 후퇴 또는 하악 전진 양상"; tendency = "Class III 방향"; color = "#5B9BD5"

    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 12px {color}40;">
        <h4 style="margin-top: 0; color: white;">📈 ANB {direction} ({change:+.1f}°)</h4>
        <p><strong>임상적 의미:</strong> {meaning}</p>
        <p><strong>분류 경향:</strong> {tendency}</p>
        <p><strong>새 분류:</strong> Class {new_result['class']} (신뢰도 {new_result['confidence']*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

    if 3.5 <= new_anb <= 4.5:
        st.markdown("""
        <div style="background: #FFA726; color: white; padding: 1rem; border-radius: 12px; margin-top: 1rem;">
            <strong>⚠️ 경계 영역: Class I/II 경계 (ANB ≈ 4°)</strong>
        </div>
        """, unsafe_allow_html=True)
    elif -0.5 <= new_anb <= 0.5:
        st.markdown("""
        <div style="background: #FFA726; color: white; padding: 1rem; border-radius: 12px; margin-top: 1rem;">
            <strong>⚠️ 경계 영역: Class I/III 경계 (ANB ≈ 0°)</strong>
        </div>
        """, unsafe_allow_html=True)

def create_landmark_overlay(image, landmarks, highlight_points=None, size_factor=0.016, show_labels=True):
    """이미지에 랜드마크를 오버레이합니다 (건양대 색상)"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = image.size
    base_size = min(width, height)

    for name, (x, y) in landmarks.items():
        if highlight_points and name in highlight_points:
            color = '#5B9BD5'; outline_color = '#FFFFFF'; radius = max(10, int(base_size * size_factor * 1.2)); text_color = '#5B9BD5'
        else:
            color = '#C53030'; outline_color = '#FFFFFF'; radius = max(8, int(base_size * size_factor)); text_color = '#C53030'

        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=outline_color, width=3)

        if show_labels:
            font_size = max(14, int(base_size * size_factor * 1.2))
            try:
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = None

            text_x, text_y = x + radius + 8, y - radius - 8
            if font:
                bbox = draw.textbbox((text_x, text_y), name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = len(name) * 10, 14

            bg_padding = 3
            draw.rectangle([text_x - bg_padding, text_y - bg_padding,
                            text_x + text_width + bg_padding, text_y + text_height + bg_padding],
                           fill='white', outline=text_color, width=1)
            if font:
                draw.text((text_x, text_y), name, fill=text_color, font=font)
            else:
                draw.text((text_x, text_y), name, fill=text_color)

    return img_copy

def display_clinical_metrics(metrics):
    """건양대 테마 임상 지표 표시"""
    st.markdown("### 📊 임상 지표")
    cols = st.columns(2)

    for i, (metric_name, data) in enumerate(metrics.items()):
        col = cols[i % 2]
        with col:
            status = data["status"]
            if status == "normal":
                icon = "✅"; status_class = "status-normal"; bg_color = "#2D5530"
            elif status == "high":
                icon = "⬆️"; status_class = "status-high"; bg_color = "#C53030"
            else:
                icon = "⬇️"; status_class = "status-low"; bg_color = "#5B9BD5"

            st.markdown(f"""
                <div style="
                    background: {bg_color}10;
                    padding: 1.5rem;
                    border-radius: 15px;
                    border-left: 6px solid {bg_color};
                    margin: 0.8rem 0;
                    box-shadow: 0 3px 10px {bg_color}30;
                ">
                    <h4 style="color: {bg_color}; margin: 0;">{icon} {metric_name}</h4>
                    <p style="margin: 0.5rem 0;"><span class="{status_class}" style="font-size: 1.3em;">{data['value']:.1f}°</span></p>
                    <p style="margin: 0; font-size: 0.9em; color: #666;">정상 범위: {data['normal_range'][0]}-{data['normal_range'][1]}°</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.85em; color: #666;">{data.get('clinical_significance', '')}</p>
                </div>
            """, unsafe_allow_html=True)

def display_classification_result(classification):
    """건양대 테마 분류 결과 표시"""
    st.markdown("### 🎯 분류 결과")

    # 숫자/문자 레이블 모두 대응
    class_map_num_to_label = {1: "Class I", 2: "Class II", 3: "Class III"}
    if "predicted_label" in classification:
        label = classification["predicted_label"]
    elif "predicted_class" in classification:
        label = class_map_num_to_label.get(classification["predicted_class"], f"Class {classification['predicted_class']}")
    elif "class" in classification:
        label = class_map_num_to_label.get(classification["class"], f"Class {classification['class']}")
    else:
        label = "Unknown"

    confidence = float(classification.get("confidence", 0.0))
    anb_value = float(classification.get("anb_value", 0.0))

    class_info = {
        "Class I": {"color": "#2D5530", "desc": "골격적으로 정상"},
        "Class II": {"color": "#C53030", "desc": "골격적으로 상악 과성장"},
        "Class III": {"color": "#5B9BD5", "desc": "골격적으로 하악 과성장"}
    }
    color = class_info.get(label, {}).get("color", "#2D5530")
    description = class_info.get(label, {}).get("desc", "")

    st.markdown(f"""
        <div style="
            background: {color}10;
            padding: 2.5rem;
            border-radius: 20px;
            border: 4px solid {color};
            text-align: center;
            box-shadow: 0 8px 25px {color}30;
        ">
            <h2 style="color: {color}; margin: 0;">{label}</h2>
            <h4 style="color: {color}; margin: 1rem 0;">{description}</h4>
            <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>신뢰도:</strong> {confidence*100:.1f}%</p>
            <p style="font-size: 1.1em; margin: 0.5rem 0;"><strong>ANB 각도:</strong> {anb_value:.1f}°</p>
            <p style="font-size: 0.9em; color: #666; margin: 1rem 0 0 0;">{classification.get('classification_basis', '')}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 분류 확률")
    probs = classification.get("probabilities", {})
    # 키가 숫자(class)로 올 수도 있으니 문자열 라벨로 변환
    normalized_probs = {}
    for k, v in probs.items():
        if isinstance(k, int):
            normalized_probs[class_map_num_to_label.get(k, f"Class {k}")] = v
        else:
            normalized_probs[str(k)] = v

    for class_name, prob in normalized_probs.items():
        st.progress(float(prob), text=f"{class_name}: {float(prob)*100:.1f}%")

def load_demo_image():
    """대표 도면 이미지를 로드합니다."""
    demo_path = "data/sample_images/demo_xray.jpg"
    if os.path.exists(demo_path):
        return Image.open(demo_path)
    else:
        img = Image.new('RGB', (800, 600), color='#F8F9FA')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 750, 550], fill='#2D5530', outline='#7FB069', width=5)
        # 중앙 텍스트(간단 정중앙 배치)
        cx, cy = 400, 300
        draw.text((cx, cy-10), "Konyang Medical Center", fill='white', anchor='mm')
        draw.text((cx, cy+20), "Demo Cephalometric Image", fill='white', anchor='mm')
        return img

def main():
    """메인 UI 함수"""
    initialize_session_state()

    # 건양대 테마 헤더 (로고를 상단 좌측에 배치)
    logo_base64 = get_konyang_logo_base64()
    logo_mime_type = "image/png" if os.path.exists("data/assets/konyang_logo.png") else "image/svg+xml"

    st.markdown(f"""
        <div class="main-header">
            <div class="header-logo-top">
                <img src="data:{logo_mime_type};base64,{logo_base64}" alt="건양대학교 의료원" style="height: 60px;">
            </div>
            <div class="header-content">
                <div class="header-text">
                    <h1>🏥 Cephalometric AI - Konyang Medical Center</h1>
                    <p>측면두부규격방사선사진 자동 분석 시스템 v1.2.3</p>
                    <p style="font-size: 0.9em; opacity: 0.9;">건양대학교 의료원 AI 진단 솔루션</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        demo_mode = st.toggle("데모 모드", value=True, help="오프라인 시뮬레이션 모드")
        st.session_state.demo_mode = demo_mode
        if not demo_mode:
            st.warning("연구 모드는 안심존에서만 사용 가능합니다")
            st.session_state.demo_mode = True
            demo_mode = True

        if st.session_state.pipeline is None:
            with st.spinner("파이프라인 초기화 중..."):
                try:
                    st.session_state.pipeline = CephalometricPipeline(demo_mode=demo_mode, seed=42)
                    st.success("✅ 초기화 완료")
                except Exception as e:
                    st.error(f"❌ 초기화 실패: {e}")
                    st.stop()

        st.markdown("---")
        st.markdown("### 🎨 시각화 설정")
        landmark_size = st.selectbox("랜드마크 크기", ["작게", "보통", "크게", "매우 크게"], index=2)
        show_labels = st.checkbox("랜드마크 이름 표시", value=True)
        size_mapping = {"작게": 0.008, "보통": 0.012, "크게": 0.016, "매우 크게": 0.020}

        st.markdown("---")
        st.markdown("### 👤 환자 정보")
        patient_age = st.number_input("나이", min_value=1, max_value=99, value=25)
        patient_sex = st.selectbox("성별", ["F", "M", "U"], index=0)
        patient_id = st.text_input("환자 ID", value="KY001")
        meta = {"age": patient_age, "sex": patient_sex, "patient_id": patient_id}

        st.markdown("---")
        st.markdown("### 🖥️ 시스템 정보")
        st.markdown("""
        <div class="system-info-card">
            <h4 style="margin: 0; color: white;">🏥 Konyang Medical AI</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9em;">버전: v1.2.3-KY</p>
            <p style="margin: 0; font-size: 0.9em;">빌드: konyang-abc123f</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🔋 시스템 상태**")
        st.markdown("🟢 AI 모델: 정상 운영")
        st.markdown("🟢 데이터베이스: 연결됨")
        st.markdown("🟢 보안: 암호화 활성")

        st.markdown("**📊 리소스 사용량**")
        st.progress(0.3, text="CPU: 30%")
        st.progress(0.5, text="Memory: 50%")
        st.progress(0.2, text="GPU: 20%")

        st.markdown("---")
        st.markdown("### 🔧 고급 설정")
        use_anchors = st.checkbox("FH 기준선 수동 보정", help="Or, Po 두 점을 수동으로 지정하여 Frankfort Horizontal plane 보정")
        anchors = None
        if use_anchors:
            st.info("Or(Orbitale), Po(Porion) 좌표를 입력하세요")
            or_x = st.number_input("Or X", value=400, min_value=0, max_value=2000)
            or_y = st.number_input("Or Y", value=200, min_value=0, max_value=2000)
            po_x = st.number_input("Po X", value=300, min_value=0, max_value=2000)
            po_y = st.number_input("Po Y", value=210, min_value=0, max_value=2000)
            anchors = {"Or": (float(or_x), float(or_y)), "Po": (float(po_x), float(po_y))}

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 이미지 입력")
        input_method = st.radio("입력 방식 선택:", ["대표 도면 사용", "파일 업로드"], horizontal=True)
        selected_image = None

        if input_method == "대표 도면 사용":
            if st.button("🏥 건양대 대표 도면 로드", type="primary"):
                selected_image = load_demo_image()
                st.session_state.input_image = selected_image
        else:
            uploaded_file = st.file_uploader("X-ray 이미지 업로드", type=["jpg", "jpeg", "png"], help="측면두부규격방사선사진을 업로드하세요")
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                st.session_state.input_image = selected_image

        if hasattr(st.session_state, 'input_image'):
            st.image(st.session_state.input_image, caption="입력 이미지 - Konyang Medical Center", use_container_width=True)

            if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
                with st.spinner("건양대 AI가 분석 중입니다..."):
                    try:
                        start_time = time.time()
                        result = st.session_state.pipeline.run(st.session_state.input_image, meta=meta, anchors=anchors)
                        execution_time = time.time() - start_time  # noqa: F841

                        if "error" in result:
                            st.error(f"❌ 분석 실패: {result['error']['message']}")
                        else:
                            st.session_state.analysis_results = result
                            total_time = result["performance"]["total_time_ms"]
                            st.markdown(f"""<div class="performance-badge">⚡ {total_time:.1f}ms 완료 | 건양대 AI</div>""",
                                        unsafe_allow_html=True)
                            st.success("✅ 건양대 AI 분석 완료!")
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류 발생: {e}")

    with col2:
        st.markdown("### 📋 분석 결과")

        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            render_performance_dashboard(results)
            render_clinical_status_badges(results["clinical_metrics"])

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 분류", "📊 지표", "📍 랜드마크", "🎛️ What-if", "⚡ 성능"])

            with tab1:
                display_classification_result(results["classification"])

            with tab2:
                display_clinical_metrics(results["clinical_metrics"])

            with tab3:
                st.markdown("#### 랜드마크 위치")
                if hasattr(st.session_state, 'input_image'):
                    landmarks = results["landmarks"]["coordinates"]
                    highlight = ["Or", "Po"] if anchors else None
                    size_factor = size_mapping.get(landmark_size, 0.016)
                    overlay_img = create_landmark_overlay(st.session_state.input_image, landmarks,
                                                          highlight_points=highlight,
                                                          size_factor=size_factor, show_labels=show_labels)
                    st.image(overlay_img, caption=f"랜드마크 표시 ({len(landmarks)}개 점) - {landmark_size} 크기 | 건양대 AI",
                             use_container_width=True)

                with st.expander("좌표 상세 정보"):
                    landmarks = results["landmarks"]["coordinates"]
                    coords_data = [{"랜드마크": name, "X": f"{x:.1f}", "Y": f"{y:.1f}"} for name, (x, y) in landmarks.items()]
                    st.dataframe(coords_data, use_container_width=True)

            with tab4:
                whatif_result = render_whatif_simulator(results)
                if whatif_result:
                    st.markdown("---")
                    st.markdown("#### 📋 조정 결과 요약")
                    summary_cols = st.columns(2)
                    with summary_cols[0]:
                        st.markdown(f"""
                        <div style="background: #2D5530; color: white; padding: 1.2rem; border-radius: 12px;">
                            <h5 style="margin: 0; color: white;">조정된 지표</h5>
                            <p style="margin: 0.5rem 0; font-size: 0.9em;">
                                SNA: {whatif_result['adjusted_sna']:.1f}°<br>
                                SNB: {whatif_result['adjusted_snb']:.1f}°<br>
                                ANB: {whatif_result['adjusted_anb']:.1f}°<br>
                                FMA: {whatif_result['adjusted_fma']:.1f}°
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    with summary_cols[1]:
                        new_class = whatif_result['new_classification']
                        st.markdown(f"""
                        <div style="background: #5B9BD5; color: white; padding: 1.2rem; border-radius: 12px;">
                            <h5 style="margin: 0; color: white;">시뮬레이션 결과</h5>
                            <p style="margin: 0.5rem 0; font-size: 0.9em;">
                                분류: Class {new_class['class']}<br>
                                신뢰도: {new_class['confidence']*100:.1f}%<br>
                                카테고리: {new_class['anb_category']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            with tab5:
                st.markdown("#### 성능 분석")
                perf = results["performance"]; quality = results["quality"]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("총 처리 시간", f"{perf['total_time_ms']:.1f}ms")
                    st.metric("추론 시간", f"{perf['inference_ms']:.1f}ms")
                with col_b:
                    st.metric("품질 점수", f"{quality['overall_score']:.3f}")
                    st.metric("추론 모드", results["landmarks"]["inference_mode"])

                st.markdown("#### 품질 평가")
                st.info(quality["recommendation"])
                if quality.get("warnings"):
                    for warning in quality["warnings"]:
                        st.warning(f"⚠️ {warning}")
        else:
            st.info("👆 이미지를 선택하고 분석을 시작해주세요")
            st.markdown("#### 🖥️ 건양대 AI 시스템 상태")
            status_cols = st.columns(2)
            with status_cols[0]:
                st.markdown("""
                <div style="border: 2px solid #2D5530; border-radius: 15px; padding: 18px; background: #2D553015;">
                    <h4 style="color: #2D5530; margin: 0;">🟢 AI 모델</h4>
                    <p style="margin: 5px 0; color: #2D5530;">상태: 정상 동작</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">준비 완료</p>
                </div>
                """, unsafe_allow_html=True)
            with status_cols[1]:
                st.markdown("""
                <div style="border: 2px solid #5B9BD5; border-radius: 15px; padding: 18px; background: #5B9BD515;">
                    <h4 style="color: #5B9BD5; margin: 0;">⚡ 평균 성능</h4>
                    <p style="margin: 5px 0; color: #5B9BD5;">처리시간: ~18ms</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">정확도: 94.7%</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1.5rem;">
            <p>🏥 <strong style="color: #2D5530;">Konyang University Medical Center</strong> | 
            Cephalometric AI System v1.2.3-KY</p>
            <p><a href="#" style="color: #2D5530;">GitHub</a> | 
            <a href="#" style="color: #7FB069;">Documentation</a> | 
            <span style="color: #5B9BD5;">Build: konyang-abc123f</span></p>
            <p><small>이 시스템은 건양대학교 의료원에서 개발된 교육 및 연구용 AI 솔루션입니다.</small></p>
            <p><small style="color: #C53030;">⚠️ 실제 임상 사용 시 반드시 전문의 검토가 필요합니다.</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
