# -*- coding: utf-8 -*-
"""
Demo Inference Engine
실제 AI 모델 없이 랜드마크를 추론하는 데모용 엔진입니다.

주요 기능:
- 대표 도면 매칭 (SHA256 해시 기반)
- 평균 형상 기반 히ュー리스틱 추론
- FH 기준 좌표 보정 (Or/Po 앵커 포인트)
- 결정론적 노이즈 추가
"""

import json
import os
import hashlib
import time
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, Any, List

# 19개 랜드마크 목록
LANDMARK_NAMES = [
    "N", "S", "Ar", "Or", "Po", "A", "B", "U1", "Ls", "Pog'",
    "Go", "Pog", "Me", "ANS", "PNS", "Gn", "L1", "Li", "Pn"
]

def load_json_config(file_path: str) -> Dict:
    """JSON 설정 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파일 파싱 오류: {file_path}, {e}")

def hash_image(pil_image: Image.Image) -> str:
    """
    이미지의 SHA256 해시를 계산합니다.
    일관성을 위해 256x256 그레이스케일로 정규화 후 계산합니다.
    """
    # 정규화: 256x256 그레이스케일
    normalized = pil_image.convert("L").resize((256, 256))
    image_bytes = normalized.tobytes()
    
    # SHA256 해시 계산
    hash_object = hashlib.sha256(image_bytes)
    return hash_object.hexdigest()

def scale_normalized_landmarks(normalized_landmarks: Dict[str, Tuple[float, float]], 
                              image_width: int, 
                              image_height: int) -> Dict[str, Tuple[float, float]]:
    """
    정규화된 좌표(0~1)를 실제 이미지 크기에 맞춰 스케일링합니다.
    """
    scaled = {}
    for name, (norm_x, norm_y) in normalized_landmarks.items():
        x = float(norm_x * image_width)
        y = float(norm_y * image_height)
        scaled[name] = (x, y)
    return scaled

def similarity_transform_2d(points: Dict[str, Tuple[float, float]], 
                           src_anchor1: Tuple[float, float], 
                           src_anchor2: Tuple[float, float],
                           dst_anchor1: Tuple[float, float], 
                           dst_anchor2: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
    """
    두 앵커 포인트를 기준으로 similarity transformation을 적용합니다.
    주로 FH plane 정렬에 사용 (Or, Po 기준)
    """
    import math
    
    # 원본 벡터
    src_dx = src_anchor2[0] - src_anchor1[0]
    src_dy = src_anchor2[1] - src_anchor1[1]
    src_dist = math.sqrt(src_dx**2 + src_dy**2)
    
    # 목표 벡터  
    dst_dx = dst_anchor2[0] - dst_anchor1[0]
    dst_dy = dst_anchor2[1] - dst_anchor1[1]
    dst_dist = math.sqrt(dst_dx**2 + dst_dy**2)
    
    if src_dist == 0 or dst_dist == 0:
        return points  # 변환 불가
    
    # 스케일 계산
    scale = dst_dist / src_dist
    
    # 회전 각도 계산
    src_angle = math.atan2(src_dy, src_dx)
    dst_angle = math.atan2(dst_dy, dst_dx)
    rotation = dst_angle - src_angle
    
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    
    # 변환 적용
    transformed = {}
    for name, (x, y) in points.items():
        # 1. 평행이동 (src_anchor1을 원점으로)
        tx = x - src_anchor1[0]
        ty = y - src_anchor1[1]
        
        # 2. 회전 + 스케일
        rx = scale * (cos_r * tx - sin_r * ty)
        ry = scale * (sin_r * tx + cos_r * ty)
        
        # 3. 평행이동 (dst_anchor1으로)
        final_x = rx + dst_anchor1[0]
        final_y = ry + dst_anchor1[1]
        
        transformed[name] = (float(final_x), float(final_y))
    
    return transformed

def add_deterministic_jitter(points: Dict[str, Tuple[float, float]], 
                           sigma: float = 1.5, 
                           seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    결정론적 노이즈를 추가합니다 (시드 고정으로 재현 가능).
    """
    rng = np.random.RandomState(seed)
    jittered = {}
    
    for name, (x, y) in points.items():
        # 가우시안 노이즈 추가
        dx, dy = rng.normal(0, sigma, 2)
        jittered[name] = (float(x + dx), float(y + dy))
    
    return jittered

def clamp_points_to_image(points: Dict[str, Tuple[float, float]], 
                         image_width: int, 
                         image_height: int) -> Dict[str, Tuple[float, float]]:
    """
    좌표가 이미지 경계를 벗어나지 않도록 클램핑합니다.
    """
    clamped = {}
    for name, (x, y) in points.items():
        clamped_x = max(0, min(x, image_width - 1))
        clamped_y = max(0, min(y, image_height - 1))
        clamped[name] = (float(clamped_x), float(clamped_y))
    
    return clamped

class DemoInference:
    """
    데모용 랜드마크 추론 엔진
    """
    
    def __init__(self, 
                 demo_config_path: str = "data/clinical_standards/demo_landmarks.json",
                 mean_shape_path: str = "data/clinical_standards/mean_shape.json",
                 seed: int = 42):
        """
        초기화
        
        Args:
            demo_config_path: 대표 도면 설정 파일 경로
            mean_shape_path: 평균 형상 파일 경로  
            seed: 난수 시드
        """
        self.seed = seed
        self.demo_config = load_json_config(demo_config_path)
        self.mean_shape = load_json_config(mean_shape_path)
        
        print(f"✅ DemoInference 초기화 완료 (seed={seed})")
    
    def load_precomputed_if_match(self, pil_image: Image.Image) -> Tuple[Optional[Dict], Optional[str]]:
        """
        이미지가 대표 도면과 일치하는지 확인하고, 일치하면 사전 계산된 좌표를 반환합니다.
        """
        image_hash = hash_image(pil_image)
        demo_hash = self.demo_config.get("image_sha256", "")
        
        if image_hash == demo_hash:
            # 정규화된 좌표를 현재 이미지 크기에 맞춰 스케일링
            width, height = pil_image.size
            landmarks = scale_normalized_landmarks(
                self.demo_config["landmarks"], width, height
            )
            return landmarks, "precomputed"
        
        return None, None
    
    def heuristic_landmarks(self, pil_image: Image.Image) -> Dict[str, Tuple[float, float]]:
        """
        평균 형상 기반으로 히ュー리스틱 랜드마크를 생성합니다.
        """
        width, height = pil_image.size
        
        # 평균 형상을 현재 이미지 크기에 스케일링
        landmarks = scale_normalized_landmarks(
            self.mean_shape["landmarks"], width, height
        )
        
        return landmarks
    
    def predict_landmarks(self, 
                         pil_image: Image.Image, 
                         anchors: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Dict[str, Tuple[float, float]], str]:
        """
        이미지에서 랜드마크를 예측합니다.
        
        Args:
            pil_image: 입력 이미지
            anchors: 사용자가 제공한 앵커 포인트 (Or, Po 등)
        
        Returns:
            (landmarks, mode) 튜플
            - landmarks: 예측된 랜드마크 좌표
            - mode: "precomputed" | "heuristic" | "manual_corrected"
        """
        start_time = time.perf_counter()
        width, height = pil_image.size
        
        # 1단계: 대표 도면 매칭 시도
        landmarks, mode = self.load_precomputed_if_match(pil_image)
        
        if landmarks is None:
            # 2단계: 히ュー리스틱 생성
            landmarks = self.heuristic_landmarks(pil_image)
            mode = "heuristic"
            
            # 3단계: 앵커 기반 보정 (Or, Po 제공 시)
            if anchors and "Or" in anchors and "Po" in anchors:
                # 현재 Or, Po 위치
                current_or = landmarks["Or"]
                current_po = landmarks["Po"]
                
                # 사용자 제공 Or, Po 위치
                target_or = anchors["Or"]
                target_po = anchors["Po"]
                
                # Similarity transformation 적용
                landmarks = similarity_transform_2d(
                    landmarks, current_or, current_po, target_or, target_po
                )
                mode = "manual_corrected"
        
        # 4단계: 결정론적 노이즈 추가
        landmarks = add_deterministic_jitter(landmarks, sigma=1.5, seed=self.seed)
        
        # 5단계: 이미지 경계 클램핑
        landmarks = clamp_points_to_image(landmarks, width, height)
        
        elapsed = time.perf_counter() - start_time
        print(f"🎯 랜드마크 예측 완료: {mode} ({elapsed*1000:.1f}ms)")
        
        return landmarks, mode
    
    def get_inference_info(self) -> Dict[str, Any]:
        """
        추론 엔진 정보를 반환합니다.
        """
        return {
            "engine": "DemoInference",
            "version": "1.0",
            "seed": self.seed,
            "demo_hash": self.demo_config.get("image_sha256", "")[:16] + "...",
            "landmark_count": len(LANDMARK_NAMES),
            "supported_modes": ["precomputed", "heuristic", "manual_corrected"]
        }

def test_demo_inference():
    """
    데모 추론 엔진 테스트
    """
    print("🧪 DemoInference 테스트")
    print("=" * 50)
    
    try:
        # 1. 엔진 초기화
        engine = DemoInference()
        
        # 2. 정보 출력
        info = engine.get_inference_info()
        print("📋 엔진 정보:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 3. 테스트용 이미지 생성 (800x600 흰색 이미지)
        test_image = Image.new("RGB", (800, 600), color="white")
        
        # 4. 랜드마크 예측 (히ュー리스틱 모드)
        print("\n🎯 히ュー리스틱 모드 테스트:")
        landmarks, mode = engine.predict_landmarks(test_image)
        print(f"   모드: {mode}")
        print(f"   랜드마크 개수: {len(landmarks)}")
        print(f"   샘플 좌표 (N): {landmarks['N']}")
        print(f"   샘플 좌표 (A): {landmarks['A']}")
        
        # 5. 앵커 보정 테스트
        print("\n🔧 앵커 보정 테스트:")
        anchors = {"Or": (450, 200), "Po": (300, 220)}
        landmarks_corrected, mode_corrected = engine.predict_landmarks(test_image, anchors=anchors)
        print(f"   보정 모드: {mode_corrected}")
        print(f"   보정된 Or: {landmarks_corrected['Or']}")
        print(f"   보정된 Po: {landmarks_corrected['Po']}")
        
        # 6. 재현성 테스트 (같은 시드로 2번 실행)
        print("\n🔄 재현성 테스트:")
        landmarks1, _ = engine.predict_landmarks(test_image)
        landmarks2, _ = engine.predict_landmarks(test_image)
        
        # N 포인트로 재현성 확인
        diff = abs(landmarks1["N"][0] - landmarks2["N"][0]) + abs(landmarks1["N"][1] - landmarks2["N"][1])
        print(f"   좌표 차이 (N): {diff:.6f} (0이면 완전 재현)")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_demo_inference()