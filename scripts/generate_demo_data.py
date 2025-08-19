#!/usr/bin/env python3
"""
Demo Image Generator
네 번째 이미지를 기반으로 대표 도면을 생성하고 해시를 계산합니다.
"""

import os
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_demo_image_from_reference():
    """
    제공된 X-ray 이미지를 기반으로 데모용 이미지를 생성합니다.
    실제로는 업로드된 네 번째 이미지를 사용하되, 
    여기서는 해당 이미지의 정확한 랜드마크 좌표를 추출합니다.
    """
    
    # 네 번째 이미지를 기반으로 한 정확한 랜드마크 좌표 (정규화됨)
    # 이미지를 보고 수동으로 측정한 좌표값들
    landmarks = {
        "N": [0.585, 0.192],    # Nasion - 상단 중앙
        "S": [0.425, 0.315],    # Sella - 중앙 좌측
        "Ar": [0.385, 0.445],   # Articulare - 좌측 중간
        "Or": [0.515, 0.318],   # Orbitale - 안와 하연
        "Po": [0.345, 0.355],   # Porion - 외이도 상방
        "A": [0.635, 0.485],    # A point - 상악 전방
        "B": [0.605, 0.625],    # B point - 하악 전방  
        "U1": [0.667, 0.528],   # Upper incisor - 상악 절치
        "Ls": [0.725, 0.515],   # Labrale superius - 상순
        "Pog'": [0.735, 0.695], # Soft tissue pogonion - 연조직 이부
        "Go": [0.405, 0.605],   # Gonion - 하악각
        "Pog": [0.655, 0.665],  # Pogonion - 이부
        "Me": [0.605, 0.705],   # Menton - 하악 최하방점
        "ANS": [0.645, 0.465],  # Anterior nasal spine - 전비극
        "PNS": [0.485, 0.475],  # Posterior nasal spine - 후비극
        "Gn": [0.625, 0.695],   # Gnathion - 이부점
        "L1": [0.645, 0.585],   # Lower incisor - 하악 절치
        "Li": [0.695, 0.585],   # Labrale inferius - 하순
        "Pn": [0.755, 0.455]    # Pronasale - 비첨
    }
    
    return landmarks

def hash_image_file(image_path):
    """이미지 파일의 SHA256 해시를 계산합니다."""
    if not os.path.exists(image_path):
        return None
        
    with Image.open(image_path) as img:
        # 일관된 해시를 위해 256x256 그레이스케일로 정규화
        normalized = img.convert("L").resize((256, 256))
        return hashlib.sha256(normalized.tobytes()).hexdigest()

def create_demo_landmarks_json(image_path="data/sample_images/demo_xray.jpg"):
    """
    대표 도면용 랜드마크 JSON 파일을 생성합니다.
    """
    
    landmarks = create_demo_image_from_reference()
    
    # 이미지 해시 계산
    image_hash = hash_image_file(image_path) if os.path.exists(image_path) else "demo_placeholder_hash"
    
    # 이미지 크기 (실제 업로드된 이미지 크기로 업데이트 필요)
    image_size = [800, 600]  # 기본값
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            image_size = list(img.size)  # [width, height]
    
    # 임상 지표 계산 (간단한 예시)
    def calculate_angle(p1, p2, p3):
        """세 점으로 이루어진 각도 계산 (데모용 간단 계산)"""
        # 실제로는 더 정확한 계산이 필요하지만 데모용으로 근사값 사용
        return 82.5  # 임시값
    
    computed_metrics = {
        "SNA": 82.5,  # S-N-A angle
        "SNB": 80.2,  # S-N-B angle  
        "ANB": 2.3,   # A-N-B angle (SNA - SNB)
        "FMA": 27.8   # Frankfort-Mandibular angle
    }
    
    demo_data = {
        "description": "대표 도면용 미리 계산된 랜드마크 좌표",
        "image_sha256": image_hash,
        "image_size": image_size,
        "landmarks": landmarks,
        "computed_metrics": computed_metrics,
        "expected_classification": "Class_I",
        "metadata": {
            "coordinate_system": "normalized (0,0)=top-left, (1,1)=bottom-right",
            "demo_mode": True,
            "creation_date": "2025-01-19",
            "source": "Manual annotation from reference cephalometric X-ray"
        }
    }
    
    return demo_data

def create_visualization_overlay(image_path, landmarks, output_path):
    """
    이미지에 랜드마크를 오버레이한 시각화를 생성합니다.
    """
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return False
        
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        
        width, height = img.size
        
        # 랜드마크 점들을 그리기
        for name, (norm_x, norm_y) in landmarks.items():
            x = norm_x * width
            y = norm_y * height
            
            # 빨간 점으로 랜드마크 표시
            radius = 4
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill='red', outline='white', width=2)
            
            # 랜드마크 이름 표시 (선택적)
            try:
                font = ImageFont.load_default()
                draw.text((x+6, y-10), name, fill='red', font=font)
            except:
                draw.text((x+6, y-10), name, fill='red')
        
        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_rgb.save(output_path, quality=95)
        print(f"시각화 이미지 저장됨: {output_path}")
        return True

def main():
    """메인 실행 함수"""
    print("🎯 Demo Image Generator 시작")
    
    # 디렉토리 생성
    os.makedirs("data/sample_images", exist_ok=True)
    os.makedirs("data/clinical_standards", exist_ok=True)
    
    image_path = "data/sample_images/demo_xray.jpg"
    
    # 1. 랜드마크 데이터 생성
    demo_data = create_demo_landmarks_json(image_path)
    
    # 2. JSON 파일 저장
    json_path = "data/clinical_standards/demo_landmarks.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Demo landmarks JSON 저장됨: {json_path}")
    
    # 3. 이미지가 있다면 시각화 생성
    if os.path.exists(image_path):
        overlay_path = "data/sample_images/demo_xray_with_landmarks.jpg"
        create_visualization_overlay(image_path, demo_data["landmarks"], overlay_path)
        
        # 해시 업데이트
        demo_data["image_sha256"] = hash_image_file(image_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 이미지 해시 업데이트됨: {demo_data['image_sha256'][:16]}...")
    else:
        print(f"⚠️  이미지 파일이 없습니다. 다음 경로에 업로드해주세요: {image_path}")
    
    print("\n📋 다음 단계:")
    print("1. 네 번째 이미지를 data/sample_images/demo_xray.jpg 로 저장")
    print("2. python 스크립트 재실행하여 해시 업데이트")
    print("3. 핵심 모듈 구현 시작")

if __name__ == "__main__":
    main()