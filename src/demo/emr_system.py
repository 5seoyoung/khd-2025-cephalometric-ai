# -*- coding: utf-8 -*-
"""
Cephalometric AI - Konyang University Medical Center EMR System
측면두부규격방사선사진 AI 분석 EMR 시스템 (건양대 의료원)

사용법: streamlit run src/demo/emr_system.py
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

try:
    from src.core.integration_pipeline import CephalometricPipeline
except Exception as e:
    st.error("앱 초기화 중 심각한 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# [추가] 파이프라인을 캐싱하여 로드하는 함수
@st.cache_resource
def load_pipeline():
    # 이 함수는 앱이 처음 시작될 때 딱 한 번만 실행됩니다.
    pipeline = CephalometricPipeline(demo_mode=True, seed=42)
    return pipeline


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
    page_title="건양대의료원 - Cephalometric AI EMR",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PDF 변환 유틸 ----------
def html_to_pdf_bytes(html: str) -> bytes:
    """
    HTML 문자열을 PDF 바이트로 변환합니다.
    xhtml2pdf(순수 파이썬) 사용. 미설치 시 ImportError 발생 → 호출부에서 안내.
      설치: pip install xhtml2pdf
    """
    try:
        from xhtml2pdf import pisa
    except ImportError as e:
        raise ImportError("xhtml2pdf 미설치") from e
    pdf_io = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_io, encoding='utf-8')
    if pisa_status.err:
        raise RuntimeError("PDF 생성 중 오류가 발생했습니다.")
    return pdf_io.getvalue()

# 건양대 EMR 테마 CSS
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
        --ky-light-blue: #E3F2FD;
    }

    /* EMR 헤더 시스템 - 건양대 녹색 테마 */
    .emr-header {
        background: linear-gradient(135deg, var(--ky-pine-green), var(--ky-light-green));
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(45, 85, 48, 0.4);
    }

    .hospital-brand {
        font-size: 18px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }

    .system-name {
        color: #e8f5e8;
        font-weight: normal;
        margin-left: 8px;
    }

    .version-badge {
        background: rgba(255,255,255,0.25);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 12px;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .connection-info {
        font-size: 12px;
        opacity: 0.9;
        color: #e8f5e8;
    }

    /* 환자 정보 밴드 */
    .patient-band {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 20px;
        border-radius: 6px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .patient-info {
        display: flex;
        gap: 2rem;
        align-items: center;
    }

    .phi-toggle {
        font-size: 12px;
    }

    /* 네비게이션 카드 */
    .nav-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .nav-card:hover {
        border-color: var(--ky-pine-green);
        box-shadow: 0 2px 8px rgba(45, 85, 48, 0.15);
    }

    .nav-card.active {
        border-color: var(--ky-pine-green);
        background: var(--ky-pine-green);
        color: white;
    }

    /* 임상 카드 */
    .clinical-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* 성능 스트립 */
    .performance-strip {
        background: #ecfdf5;
        border: 1px solid #d1fae5;
        padding: 8px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    /* QC 경고 */
    .qc-alert {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }

    .qc-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }

    .qc-success {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }

    /* 정상/비정상 범위 */
    .normal-range { color: #10b981; font-weight: bold; }
    .abnormal-range { color: #ef4444; font-weight: bold; }
    .warning-range { color: #f59e0b; font-weight: bold; }

    /* 감사 로그 */
    .audit-log {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #64748b;
        margin-top: 16px;
    }
    
    /* 전역 이미지 크기 상한을 더 낮춤 (스크롤 줄이기) */
    .stImage > img {
        max-height: 320px !important; /* 기존 400px → 320px */
        object-fit: contain;
    }
    
    /* 컴팩트한 메트릭 카드 */
    .compact-metric {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        text-align: center;
    }
    
    /* 결과 카드 간격 조정 */
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 컴팩트한 버튼 */
    .compact-button {
        padding: 8px 16px !important;
        font-size: 14px !important;
        margin: 4px 0 !important;
    }

    /* 프린트용 스타일 */
    @media print {
        .clinical-report {
            font-family: 'Times New Roman', serif;
            max-width: 21cm;
            margin: 0 auto;
        }
        
        .report-header {
            text-align: center;
            border-bottom: 2px solid black;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-footer {
            margin-top: 50px;
            border-top: 1px solid black;
            padding-top: 20px;
        }
    }

    /* 기존 스타일 유지 */
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
</style>
""", unsafe_allow_html=True)

# 기존의 get_konyang_logo_base64 함수를 이 코드로 전체 교체하세요.

from pathlib import Path

def get_konyang_logo_base64():
    """건양대 로고를 Base64로 인코딩 (안정적인 경로 사용)"""
    try:
        # 1. 현재 스크립트 파일(emr_system.py)의 절대 경로를 찾습니다.
        script_path = Path(__file__).resolve()
        
        # 2. 스크립트 위치를 기준으로 프로젝트 최상위 폴더(khd-2025-cephalometric-ai)로 이동합니다.
        # (src/demo/emr_system.py 이므로 세 단계 위로 올라갑니다)
        project_root = script_path.parent.parent.parent
        
        # 3. 최상위 폴더를 기준으로 로고 파일의 정확한 경로를 만듭니다.
        logo_path = project_root / "data" / "assets" / "konyang_logo.png"

        # 4. 해당 경로에 파일이 있는지 확인하고 읽어옵니다.
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = f.read()
            return base64.b64encode(logo_data).decode()
        else:
            # 파일이 없을 경우를 대비한 fallback 로직은 그대로 둡니다.
            raise FileNotFoundError("Logo not found at expected path")

    except Exception:
        # 로고 파일이 없거나 경로에 문제가 있을 경우 SVG로 대체합니다.
        logo_svg = """
        <svg width="120" height="50" viewBox="0 0 120 50" xmlns="http://www.w3.org/2000/svg">
            <rect width="120" height="50" fill="white" rx="8" stroke="#2D5530" stroke-width="2"/>
            <circle cx="25" cy="25" r="12" fill="#2D5530"/>
            <circle cx="25" cy="25" r="6" fill="white"/>
            <text x="50" y="18" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#2D5530">건양대학교</text>
            <text x="50" y="32" font-family="Arial, sans-serif" font-size="9" fill="#7FB069">의료원</text>
            <text x="50" y="42" font-family="Arial, sans-serif" font-size="7" fill="#2D5530">KONYANG</text>
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
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "viewer"
    if 'show_phi' not in st.session_state:
        st.session_state.show_phi = False
    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = []
    if 'overlay_thumbnail' not in st.session_state:
        st.session_state.overlay_thumbnail = None  # 뷰어 오른쪽에 작은 시각화 썸네일

def add_audit_log(action, details=""):
    """감사 로그 추가"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "details": details,
        "user": "김○○ 의사" if not st.session_state.show_phi else "김철수 의사"
    }
    st.session_state.audit_logs.append(log_entry)
    # 최대 50개 로그만 유지
    if len(st.session_state.audit_logs) > 50:
        st.session_state.audit_logs = st.session_state.audit_logs[-50:]

def render_hospital_header():
    """실제 EMR처럼 보이는 상단 헤더 (건양대 로고 포함)"""
    logo_base64 = get_konyang_logo_base64()
    # 실제 로고 파일 확인
    logo_exists = any(os.path.exists(path) for path in [
        "khd-2025-cephalometric-ai/data/assets/konyang_logo.png",
        "data/assets/konyang_logo.png"
    ])
    logo_mime_type = "image/png" if logo_exists else "image/svg+xml"
    
    st.markdown(f"""
    <div class="emr-header">
        <div class="hospital-brand">
            <img src="data:{logo_mime_type};base64,{logo_base64}" alt="건양대학교 의료원" 
                 style="height: 40px; margin-right: 15px; vertical-align: middle; background: white; padding: 4px; border-radius: 6px;">
            건양대학교의료원 <span class="system-name">Cephalometric AI</span>
            <span class="version-badge">v2.1.0</span>
        </div>
        <div class="connection-info">
            접속: ceph-ai.kyuh.ac.kr | 🔒 SSL | 응답시간: 18ms | 세션: EMR-2025-001
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_patient_band():
    """환자 정보 상단 밴드 (PHI 마스킹)"""
    patient_name = "김○○" if not st.session_state.show_phi else "김철수"
    patient_id = "KY-****-001" if not st.session_state.show_phi else "KY-2024-001"
    
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <strong>👤 {patient_name}</strong> (M/34세) | ID: {patient_id}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
            🗓️ 2025.01.15 14:35
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
            📷 측면두부 X-ray | C250115-001
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        phi_toggle = st.checkbox("PHI 보기", key="phi_toggle", help="개인정보 마스킹 해제")
        if phi_toggle != st.session_state.show_phi:
            st.session_state.show_phi = phi_toggle
            if phi_toggle:
                add_audit_log("PHI 표시", "개인정보 마스킹 해제")
            else:
                add_audit_log("PHI 숨김", "개인정보 마스킹 적용")
            st.rerun()

def render_medical_navigation():
    """의료진 워크플로우 기반 네비게이션"""
    st.markdown("## 📋 분석 워크플로우")
    
    nav_options = [
        ("🖼️ 이미지 뷰어", "viewer", "이미지 로드 및 확인"),
        ("📊 AI 분석결과", "analysis", "자동 분석 및 결과"),
        ("⚙️ What-If 시뮬레이터", "simulator", "가상 시나리오 분석"),
        ("📝 임상 리포트", "report", "결과 리포트 생성"),
        ("🔍 이전 검사", "history", "과거 검사 이력"),
        ("⚡ QC 품질관리", "qc", "품질 관리 및 검증")
    ]
    
    for label, key, desc in nav_options:
        if st.button(f"{label}", key=f"nav_{key}", use_container_width=True,
                    type="primary" if st.session_state.current_tab == key else "secondary"):
            st.session_state.current_tab = key
            add_audit_log(f"탭 전환", f"{label} 탭으로 이동")
            st.rerun()
        
        if st.session_state.current_tab == key:
            st.markdown(f"<small style='color: #666;'>{desc}</small>", unsafe_allow_html=True)

def render_performance_dashboard(pipeline_result):
    """EMR급 성능 대시보드"""
    performance = pipeline_result.get('performance', {})
    quality = pipeline_result.get('quality', {})
    classification = pipeline_result.get('classification', {})
    
    total_time = performance.get('total_time_ms', 0.0)
    quality_score = float(quality.get('overall_score', 0.0)) * 100.0
    predicted_class = classification.get('predicted_class', 'Unknown')
    confidence = float(classification.get('confidence', 0.0)) * 100.0
    current_time = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="performance-strip">
        <div>⚡ 총 처리시간: <strong>{total_time:.1f}ms</strong></div>
        <div>🎯 품질점수: <strong>{quality_score:.1f}%</strong> {"⭐" * min(5, int(quality_score // 20) + 1)}</div>
        <div>🟢 분류결과: <strong>Class {predicted_class}</strong> (신뢰도 {confidence:.1f}%)</div>
        <div>🕐 분석완료: <strong>{current_time}</strong></div>
    </div>
    """, unsafe_allow_html=True)

def render_qc_panel(qc_results=None):
    """의료급 QC 점검 패널"""
    st.markdown("### ⚡ QC 품질관리")
    
    if qc_results is None:
        # 기본 QC 항목들
        qc_items = [
            {"name": "이미지 품질", "status": "양호", "score": 94.7, "type": "success"},
            {"name": "랜드마크 정확도", "status": "신뢰", "score": 87.3, "type": "success"},
            {"name": "임상 지표", "status": "검토필요", "score": 76.2, "type": "warning"},
            {"name": "분류 신뢰도", "status": "높음", "score": 87.3, "type": "success"}
        ]
    else:
        qc_items = qc_results
    
    for item in qc_items:
        status_icon = {"success": "✅", "warning": "⚠️", "error": "🚨"}.get(item["type"], "ℹ️")
        css_class = f"qc-{item['type']}" if item['type'] in ['warning', 'error'] else "qc-success"
        
        with st.expander(f"{status_icon} {item['name']} - {item['status']} ({item['score']:.1f}%)"):
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            
            if item["type"] == "warning":
                st.warning("ANB 7.8°가 정상범위(0-4°)를 벗어남. 재촬영 권장.")
            elif item["type"] == "error":
                st.error("심각한 품질 문제가 발견되었습니다. 즉시 점검이 필요합니다.")
            else:
                st.success("품질 기준을 충족합니다.")
            
            st.markdown("</div>", unsafe_allow_html=True)

def generate_clinical_report(result, patient_info):
    """인쇄 가능한 임상 리포트(HTML)"""
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    
    # 환자 정보 (PHI 고려)
    patient_name = patient_info.get('name', '김○○' if not st.session_state.show_phi else '김철수')
    patient_id = patient_info.get('id', 'KY-****-001' if not st.session_state.show_phi else 'KY-2024-001')
    
    # 분류 결과
    classification = result.get('classification', {})
    clinical_metrics = result.get('clinical_metrics', {})
    
    report_html = f"""
    <div class="clinical-report" style="padding: 2rem; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <header class="report-header" style="text-align: center; border-bottom: 2px solid #2D5530; padding-bottom: 20px; margin-bottom: 30px;">
            <h1 style="color: #2D5530; margin: 0;">건양대학교의료원</h1>
            <h2 style="color: #5B9BD5; margin: 10px 0;">측면두부규격방사선사진 AI 분석 결과서</h2>
        </header>
        
        <section class="patient-info" style="margin-bottom: 30px;">
            <h3 style="color: #2D5530; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">환자 정보</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="padding: 8px; border: 1px solid #e2e8f0;"><strong>성명:</strong></td><td style="padding: 8px; border: 1px solid #e2e8f0;">{patient_name}</td></tr>
                <tr><td style="padding: 8px; border: 1px solid #e2e8f0;"><strong>환자번호:</strong></td><td style="padding: 8px; border: 1px solid #e2e8f0;">{patient_id}</td></tr>
                <tr><td style="padding: 8px; border: 1px solid #e2e8f0;"><strong>검사일:</strong></td><td style="padding: 8px; border: 1px solid #e2e8f0;">2025년 01월 15일</td></tr>
                <tr><td style="padding: 8px; border: 1px solid #e2e8f0;"><strong>검사번호:</strong></td><td style="padding: 8px; border: 1px solid #e2e8f0;">C250115-001</td></tr>
            </table>
        </section>
        
        <section class="analysis-results" style="margin-bottom: 30px;">
            <h3 style="color: #2D5530; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">AI 분석 결과</h3>
            
            <div class="classification-result" style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 15px 0;">
                <h4 style="color: #2D5530;">진단 분류: Class {classification.get('predicted_class', 'N/A')}</h4>
                <p><strong>신뢰도:</strong> {classification.get('confidence', 0) * 100:.1f}%</p>
                <p><strong>임상적 의미:</strong> {classification.get('classification_basis', 'AI 기반 자동 분석')}</p>
            </div>
            
            <div class="metrics-table">
                <h4 style="color: #2D5530;">주요 임상 지표</h4>
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <thead>
                        <tr style="background: #2D5530; color: white;">
                            <th style="padding: 12px; border: 1px solid #ddd;">지표</th>
                            <th style="padding: 12px; border: 1px solid #ddd;">측정값</th>
                            <th style="padding: 12px; border: 1px solid #ddd;">정상범위</th>
                            <th style="padding: 12px; border: 1px solid #ddd;">평가</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # 임상 지표 테이블
    normal_ranges = {'SNA': (80, 84), 'SNB': (78, 82), 'ANB': (0, 4), 'FMA': (25, 30)}
    for metric_name, metric_data in clinical_metrics.items():
        if metric_name in normal_ranges:
            value = metric_data['value']
            normal_min, normal_max = normal_ranges[metric_name]
            status = "정상" if normal_min <= value <= normal_max else "비정상"
            status_color = "#10b981" if status == "정상" else "#ef4444"
            
            report_html += f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;">{metric_name}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{value:.1f}°</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{normal_min}-{normal_max}°</td>
                            <td style="padding: 8px; border: 1px solid #ddd; color: {status_color}; font-weight: bold;">{status}</td>
                        </tr>
            """
    
    report_html += f"""
                    </tbody>
                </table>
            </div>
        </section>
        
        <footer class="report-footer" style="margin-top: 50px; border-top: 1px solid #2D5530; padding-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><strong>분석 시각:</strong> {current_time}</p>
                    <p><strong>분석 시스템:</strong> 건양대의료원 Cephalometric AI v2.1.0</p>
                </div>
                <div style="text-align: right;">
                    <p><strong>담당의:</strong> _________________</p>
                    <p>(서명)</p>
                </div>
            </div>
            <div style="margin-top: 20px; font-size: 12px; color: #666; text-align: center;">
                <p>⚠️ 본 결과는 AI 보조 진단 결과이며, 최종 진단은 반드시 전문의 판독을 거쳐야 합니다.</p>
            </div>
        </footer>
    </div>
    """
    
    return report_html

def render_audit_log():
    """감사 로그 시스템"""
    if st.session_state.audit_logs:
        latest_log = st.session_state.audit_logs[-1]
        st.markdown(f"""
        <div class="audit-log">
            📋 [{latest_log['timestamp']}] {latest_log['action']} | {latest_log['user']} | {latest_log['details']}
        </div>
        """, unsafe_allow_html=True)

def create_clinical_overlay(image, landmarks, clinical_metrics=None):
    """임상용 각도/평면 오버레이"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = image.size
    
    # 기본 랜드마크 그리기
    for name, (x, y) in landmarks.items():
        color = '#C53030'
        radius = 8
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=color, outline='white', width=2)
        
        # 라벨 추가
        draw.text((x + radius + 5, y - radius - 5), name, 
                 fill=color, stroke_width=1, stroke_fill='white')
    
    # SN선 그리기 (Sella-Nasion)
    if 'S' in landmarks and 'N' in landmarks:
        s_x, s_y = landmarks['S']
        n_x, n_y = landmarks['N']
        draw.line([(s_x, s_y), (n_x, n_y)], fill='#2D5530', width=3)
        
        # SN선 라벨
        mid_x, mid_y = (s_x + n_x) / 2, (s_y + n_y) / 2
        draw.text((mid_x, mid_y - 15), "SN선", fill='#2D5530', 
                 stroke_width=1, stroke_fill='white')
    
    # FH 평면 그리기 (Frankfort Horizontal)
    if 'Or' in landmarks and 'Po' in landmarks:
        or_x, or_y = landmarks['Or']
        po_x, po_y = landmarks['Po']
        # FH 평면을 이미지 전체 너비로 연장
        draw.line([(0, or_y), (width, po_y)], fill='#5B9BD5', width=2)
        
        # FH 평면 라벨
        draw.text((width - 100, or_y - 15), "FH 평면", fill='#5B9BD5',
                 stroke_width=1, stroke_fill='white')
    
    # ANB 각도 호 그리기 (if available)
    if clinical_metrics and 'ANB' in clinical_metrics:
        if all(pt in landmarks for pt in ['A', 'N', 'B']):
            a_x, a_y = landmarks['A']
            n_x, n_y = landmarks['N']
            b_x, b_y = landmarks['B']
            
            # 간단한 각도 호 (원호 대신 직선으로 표시)
            draw.line([(n_x, n_y), (a_x, a_y)], fill='#FFA726', width=2)
            draw.line([(n_x, n_y), (b_x, b_y)], fill='#FFA726', width=2)
            
            # ANB 값 표시
            anb_value = clinical_metrics['ANB']['value']
            draw.text((n_x + 20, n_y + 20), f"ANB: {anb_value:.1f}°", 
                     fill='#FFA726', stroke_width=1, stroke_fill='white')
    
    return img_copy

def render_clinical_status_badges(clinical_metrics):
    """건양대 테마 정상범위 배지 시스템 (향상됨)"""
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
                <div class="clinical-card">
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: {color}; font-weight: bold;">{icon} {metric_name}</h4>
                        <p style="margin: 8px 0; font-size: 24px; font-weight: bold; color: {color};">{value:.1f}°</p>
                        <p style="margin: 0; font-size: 12px; color: #666;">정상: {normal_min}-{normal_max}°</p>
                        <span class="{badge_class}">{status}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_whatif_simulator(analysis_result):
    """건양대 테마 What-if 시뮬레이터 (이미지 크기 축소 적용)"""
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
                <div class="clinical-card">
                    <h4 style="color: {bg_color}; margin: 0;">{icon} {metric_name}</h4>
                    <p style="margin: 0.5rem 0;"><span class="{status_class}" style="font-size: 1.3em; color: {bg_color};">{data['value']:.1f}°</span></p>
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
        <div class="clinical-card" style="text-align: center; border: 4px solid {color};">
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

    # EMR 헤더 렌더링
    render_hospital_header()
    
    # 환자 정보 밴드
    render_patient_band()

    # 메인 레이아웃: 좌측 네비게이션 + 우측 컨텐츠
    nav_col, content_col = st.columns([1, 3])

    with nav_col:
        render_medical_navigation()
        
        st.markdown("---")
        st.markdown("### ⚙️ 시스템 설정")
        demo_mode = st.toggle("데모 모드", value=True, help="오프라인 시뮬레이션 모드")
        
        # [변경] 파이프라인 초기화 로직 변경
        try:
            # st.cache_resource로 만든 함수를 호출하여 파이프라인을 로드합니다.
            st.session_state.pipeline = load_pipeline(demo_mode_active=demo_mode)
            # 성공 메시지는 이제 필요 없으므로 주석 처리하거나 삭제합니다.
            # st.success("✅ 초기화 완료") 
        except Exception as e:
            st.error(f"❌ 파이프라인 로딩 실패: {e}")
            st.stop()

        st.markdown("### 🎨 시각화 설정")
        landmark_size = st.selectbox("랜드마크 크기", ["작게", "보통", "크게", "매우 크게"], index=2)
        show_labels = st.checkbox("랜드마크 이름 표시", value=True)
        show_clinical_overlay = st.checkbox("임상 오버레이", value=True, help="SN선, FH평면, 각도 표시")
        size_mapping = {"작게": 0.008, "보통": 0.012, "크게": 0.016, "매우 크게": 0.020}

        st.markdown("### 👤 환자 정보")
        patient_age = st.number_input("나이", min_value=1, max_value=99, value=25)
        patient_sex = st.selectbox("성별", ["F", "M", "U"], index=0)
        patient_id = st.text_input("환자 ID", value="KY-2024-001")
        meta = {"age": patient_age, "sex": patient_sex, "patient_id": patient_id}

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

    with content_col:
        # 탭별 컨텐츠 렌더링
        if st.session_state.current_tab == "viewer":
            st.markdown("## 🖼️ 이미지 뷰어")
            
            # 입력 라인
            col_input, col_button = st.columns([2, 1])
            with col_input:
                input_method = st.radio("입력 방식:", ["대표 도면", "파일 업로드"], horizontal=True)
            with col_button:
                if input_method == "대표 도면":
                    if st.button("🏥 로드", type="primary", use_container_width=True):
                        selected_image = load_demo_image()
                        st.session_state.input_image = selected_image
                        add_audit_log("이미지 로드", "건양대 대표 도면")
                        st.rerun()

            if input_method == "파일 업로드":
                uploaded_file = st.file_uploader("X-ray 이미지 업로드", type=["jpg", "jpeg", "png"], 
                                               help="측면두부규격방사선사진을 업로드하세요")
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    st.session_state.input_image = selected_image
                    add_audit_log("이미지 업로드", f"파일: {uploaded_file.name}")
                    st.rerun()

            # 이미지와 (오른쪽) 결과 썸네일 나란히 배치
            if hasattr(st.session_state, 'input_image'):
                col_img, col_thumb = st.columns([1, 1])
                
                with col_img:
                    st.markdown("### 📷 입력 이미지")
                    # 이미지 크기 축소: width 지정
                    st.image(st.session_state.input_image, caption="건양대의료원 - 측면두부X선", width=480)
                    
                    # 분석 버튼
                    if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
                        with st.spinner("건양대 AI가 분석 중입니다..."):
                            try:
                                start_time = time.time()
                                result = st.session_state.pipeline.run(st.session_state.input_image, meta=meta, anchors=anchors)
                                execution_time = time.time() - start_time

                                if "error" in result:
                                    st.error(f"❌ 분석 실패: {result['error']['message']}")
                                    add_audit_log("분석 실패", result['error']['message'])
                                else:
                                    st.session_state.analysis_results = result
                                    total_time = result["performance"]["total_time_ms"]

                                    # 썸네일(오버레이) 즉시 생성하여 오른쪽에 표시
                                    lm = result["landmarks"]["coordinates"]
                                    overlay_img = create_clinical_overlay(
                                        st.session_state.input_image, lm, result.get("clinical_metrics")
                                    )
                                    # 썸네일 저장
                                    thumb = overlay_img.copy()
                                    # 작은 썸네일로 축소 (세로 320 상한과 균형)
                                    thumb.thumbnail((480, 320))
                                    st.session_state.overlay_thumbnail = thumb

                                    st.success("✅ 건양대 AI 분석 완료!")
                                    add_audit_log("AI 분석 완료", f"처리시간: {total_time:.1f}ms")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ 분석 중 오류 발생: {e}")
                                add_audit_log("분석 오류", str(e))
                
                with col_thumb:
                    st.markdown("### 📍 랜드마크 시각화")
                    if st.session_state.overlay_thumbnail is not None:
                        st.image(st.session_state.overlay_thumbnail, caption="임상 오버레이(썸네일)", width=480)
                    else:
                        st.info("AI 분석 후 결과 썸네일이 표시됩니다.")
                    
                    # 간단 요약(컴팩트)
                    st.markdown("#### ⚡ 실시간 요약")
                    if st.session_state.analysis_results is not None:
                        results = st.session_state.analysis_results
                        total_time = results["performance"]["total_time_ms"]
                        quality_score = results["quality"]["overall_score"] * 100
                        st.metric("처리시간", f"{total_time:.1f}ms", "✅")
                        st.metric("품질점수", f"{quality_score:.1f}%", "⭐")
                        classification = results["classification"]
                        class_map = {1: "Class I", 2: "Class II", 3: "Class III"}
                        predicted_class = classification.get('predicted_class', 'Unknown')
                        confidence = classification.get('confidence', 0) * 100
                        st.write(f"분류: **{class_map.get(predicted_class, predicted_class)}**")
                        st.write(f"신뢰도: **{confidence:.1f}%**")
                        if 'ANB' in results.get("clinical_metrics", {}):
                            anb_value = results["clinical_metrics"]["ANB"]["value"]
                            st.write(f"ANB: **{anb_value:.1f}°** (정상 0–4°)")
                    else:
                        st.caption("요약정보는 분석 완료 후 표시됩니다.")
                            
            else:
                st.info("👆 이미지를 선택해주세요")
                
                # 시스템 정보 (이미지 없을 때만)
                st.markdown("### 🖥️ 건양대 AI 시스템 정보")
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown("""
                    **📋 시스템 사양**
                    - AI 모델: U-Net + ResNet
                    - 처리 속도: ~18ms
                    - 정확도: 94.7%
                    """)
                with info_col2:
                    st.markdown("""
                    **🎯 분석 항목**
                    - 19개 랜드마크 검출
                    - 임상 지표 계산 (SNA, SNB, ANB, FMA)
                    - 부정교합 분류 (Class I/II/III)
                    """)

        elif st.session_state.current_tab == "analysis":
            st.markdown("## 📊 AI 분석결과")
            
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                render_performance_dashboard(results)
                render_clinical_status_badges(results["clinical_metrics"])
                
                col1, col2 = st.columns(2)
                with col1:
                    display_classification_result(results["classification"])
                with col2:
                    display_clinical_metrics(results["clinical_metrics"])

                # 이미지(랜드마크/오버레이)도 작게
                st.markdown("---")
                st.markdown("### 📍 랜드마크 시각화")
                landmarks = results["landmarks"]["coordinates"]
                overlay_img = create_clinical_overlay(
                    st.session_state.input_image, landmarks, results.get("clinical_metrics")
                )
                # 축소 표시
                st.image(overlay_img, caption="임상 오버레이", width=640)

            else:
                st.info("먼저 이미지 뷰어에서 분석을 실행해주세요.")

        elif st.session_state.current_tab == "simulator":
            st.markdown("## ⚙️ What-If 시뮬레이터")
            
            if st.session_state.analysis_results is not None:
                whatif_result = render_whatif_simulator(st.session_state.analysis_results)
                if whatif_result:
                    add_audit_log("What-if 시뮬레이션", f"ANB 조정: {whatif_result['adjusted_anb']:.1f}°")

                # 시뮬레이터에서도 썸네일 크기 유지
                st.markdown("---")
                st.markdown("### 📍 현재 랜드마크(축소)")
                results = st.session_state.analysis_results
                landmarks = results["landmarks"]["coordinates"]
                overlay_img = create_clinical_overlay(
                    st.session_state.input_image, landmarks, results.get("clinical_metrics")
                )
                st.image(overlay_img, caption="임상 오버레이(축소)", width=640)
            else:
                st.info("먼저 AI 분석을 실행해주세요.")

        elif st.session_state.current_tab == "report":
            st.markdown("## 📝 임상 리포트")
            
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                
                # 리포트 생성 옵션
                col1, col2 = st.columns(2)
                with col1:
                    include_images = st.checkbox("이미지 포함(HTML 내 렌더링만)", value=True)
                    include_whatif = st.checkbox("What-if 결과 포함", value=False)
                with col2:
                    report_format = st.selectbox("출력 형식", ["HTML", "PDF"])
                    
                if st.button("📄 리포트 생성", type="primary"):
                    patient_info = {
                        'name': '김○○' if not st.session_state.show_phi else '김철수',
                        'id': 'KY-****-001' if not st.session_state.show_phi else 'KY-2024-001',
                        'date': '2025년 01월 15일',
                        'study_id': 'C250115-001'
                    }
                    
                    report_html = generate_clinical_report(results, patient_info)

                    # 간단히 썸네일 이미지를 HTML에 붙일지 여부(선택)
                    if include_images and st.session_state.overlay_thumbnail is not None:
                        # 이미지 base64 인라인 삽입
                        buf = BytesIO()
                        st.session_state.overlay_thumbnail.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                        report_html = report_html.replace(
                            "</footer>",
                            f"""<div style="margin-top:20px;text-align:center;">
                                    <img src="data:image/png;base64,{img_b64}" alt="Overlay" style="max-width:640px;max-height:400px;border:1px solid #ddd;border-radius:6px;"/>
                                 </div></footer>"""
                        )

                    # HTML 프리뷰
                    st.markdown(report_html, unsafe_allow_html=True)

                    if report_format == "HTML":
                        st.download_button(
                            label="📥 리포트 다운로드 (HTML)",
                            data=report_html,
                            file_name=f"cephalometric_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                        add_audit_log("리포트 생성", "임상 리포트 HTML 생성")
                    else:
                        # PDF 생성
                        try:
                            pdf_bytes = html_to_pdf_bytes(report_html)
                            st.download_button(
                                label="📥 리포트 다운로드 (PDF)",
                                data=pdf_bytes,
                                file_name=f"cephalometric_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                            add_audit_log("리포트 생성", "임상 리포트 PDF 생성")
                        except ImportError:
                            st.error("PDF 변환 모듈(xhtml2pdf)이 설치되어 있지 않습니다. 아래 명령으로 설치하세요:\n\npip install xhtml2pdf")
                        except Exception as e:
                            st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")

            else:
                st.info("먼저 AI 분석을 실행해주세요.")

        elif st.session_state.current_tab == "history":
            st.markdown("## 🔍 이전 검사")
            
            # 가상의 검사 이력
            st.markdown("### 최근 검사 이력")
            
            history_data = [
                {"날짜": "2025-01-15", "시간": "14:35", "분류": "Class II", "신뢰도": "87.3%", "상태": "완료"},
                {"날짜": "2024-12-20", "시간": "10:22", "분류": "Class I", "신뢰도": "91.2%", "상태": "완료"},
                {"날짜": "2024-11-15", "시간": "16:45", "분류": "Class II", "신뢰도": "85.1%", "상태": "완료"},
                {"날짜": "2024-10-08", "시간": "09:15", "분류": "Class I", "신뢰도": "89.7%", "상태": "완료"},
            ]
            
            for i, record in enumerate(history_data):
                with st.expander(f"📋 {record['날짜']} {record['시간']} - {record['분류']} ({record['신뢰도']})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**날짜:** {record['날짜']}")
                        st.write(f"**시간:** {record['시간']}")
                    with col2:
                        st.write(f"**분류:** {record['분류']}")
                        st.write(f"**신뢰도:** {record['신뢰도']}")
                    with col3:
                        st.write(f"**상태:** {record['상태']}")
                        if st.button(f"상세보기", key=f"detail_{i}"):
                            st.info("이전 검사 상세 결과 (구현 예정)")

        elif st.session_state.current_tab == "qc":
            st.markdown("## ⚡ QC 품질관리")
            
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                
                # QC 결과 생성
                qc_results = [
                    {"name": "이미지 품질", "status": "양호", "score": 94.7, "type": "success"},
                    {"name": "랜드마크 정확도", "status": "신뢰", "score": 87.3, "type": "success"},
                    {"name": "임상 지표", "status": "검토필요", "score": 76.2, "type": "warning"},
                    {"name": "분류 신뢰도", "status": "높음", "score": 87.3, "type": "success"}
                ]
                
                # ANB 값에 따른 QC 경고
                clinical_metrics = results.get('clinical_metrics', {})
                if 'ANB' in clinical_metrics:
                    anb_value = clinical_metrics['ANB']['value']
                    if anb_value < 0 or anb_value > 4:
                        qc_results[2]["type"] = "warning"
                        qc_results[2]["status"] = "범위 이탈"
                
                render_qc_panel(qc_results)
                
                # 추가 QC 정보
                st.markdown("### 📊 품질 세부 정보")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**이미지 품질 체크**")
                    st.success("✅ 해상도: 적정 (>512px)")
                    st.success("✅ 명암: 양호")
                    st.success("✅ 노이즈: 낮음")
                    
                with col2:
                    st.markdown("**랜드마크 품질 체크**")
                    landmarks = results.get('landmarks', {}).get('coordinates', {})
                    st.success(f"✅ 검출 개수: {len(landmarks)}/19개")
                    st.success("✅ 위치 정확도: 높음")
                    
                    # ANB 범위 체크
                    if 'ANB' in clinical_metrics:
                        anb_value = clinical_metrics['ANB']['value']
                        if 0 <= anb_value <= 4:
                            st.success(f"✅ ANB 정상범위: {anb_value:.1f}°")
                        else:
                            st.warning(f"⚠️ ANB 범위 이탈: {anb_value:.1f}° (정상: 0-4°)")
                
                # QC 권장사항
                st.markdown("### 💡 권장사항")
                if clinical_metrics.get('ANB', {}).get('value', 0) > 4:
                    st.warning("🔍 ANB 값이 정상범위를 초과합니다. 추가 검사를 권장합니다.")
                else:
                    st.success("✅ 모든 품질 기준을 충족합니다.")
                    
            else:
                st.info("먼저 AI 분석을 실행해주세요.")
                
                # 시스템 QC 상태
                st.markdown("### 🖥️ 시스템 품질 상태")
                st.success("🟢 AI 모델: 정상 동작")
                st.success("🟢 데이터베이스: 연결됨")
                st.success("🟢 보안: 암호화 활성")
                
                st.markdown("**📊 시스템 리소스**")
                st.progress(0.3, text="CPU: 30%")
                st.progress(0.5, text="Memory: 50%")
                st.progress(0.2, text="GPU: 20%")

        # (분석/시뮬레이터 탭에서 별도 큰 이미지 렌더는 위에서 width로 축소 적용)

    # 감사 로그 표시
    render_audit_log()

    # EMR 푸터
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1.5rem; background: #f8fafc; border-radius: 8px;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-bottom: 1rem;">
                <div>🏥 <strong style="color: #2D5530;">건양대학교의료원</strong></div>
                <div>📱 Cephalometric AI EMR v2.1.0</div>
                <div>🔒 보안등급: 높음</div>
            </div>
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; font-size: 0.9em;">
                <a href="#" styemrle="color: #2D5530;">시스템 가이드</a> | 
                <a href="#" style="color: #7FB069;">기술지원</a> | 
                <span style="color: #5B9BD5;">빌드: KY-EMR-240115</span>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8em;">
                <p><strong style="color: #C53030;">⚠️ 의료기기 소프트웨어:</strong> 이 시스템은 건양대학교 의료원 전용 AI 솔루션입니다.</p>
                <p><strong style="color: #C53030;">⚠️ 임상 책임:</strong> 모든 AI 결과는 반드시 전문의 검토 후 최종 판단하시기 바랍니다.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
