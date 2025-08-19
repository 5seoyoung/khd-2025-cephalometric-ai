# -*- coding: utf-8 -*-
"""
Clinical Metrics Calculator
측면두부규격방사선사진의 랜드마크로부터 임상 지표를 계산합니다.

주요 기능:
- SNA, SNB, ANB, FMA 각도 계산
- Frankfort Horizontal Plane (FH) 기반 계산
- 정상 범위와 비교하여 상태 판정
"""

import math
import json
import os
from typing import Dict, Tuple, Optional, Any

# 19개 랜드마크 정의
LANDMARK_NAMES = [
    "N", "S", "Ar", "Or", "Po", "A", "B", "U1", "Ls", "Pog'",
    "Go", "Pog", "Me", "ANS", "PNS", "Gn", "L1", "Li", "Pn"
]

def load_normal_ranges(config_path: str = "data/clinical_standards/normal_ranges.json") -> Dict:
    """정상 범위 설정을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본값 반환
        return {
            "metrics": {
                "SNA": {"normal_range": [80, 84], "unit": "degrees"},
                "SNB": {"normal_range": [78, 82], "unit": "degrees"},
                "ANB": {"normal_range": [0, 4], "unit": "degrees"},
                "FMA": {"normal_range": [25, 30], "unit": "degrees"}
            }
        }

def calculate_angle_from_three_points(p1: Tuple[float, float], 
                                     p2: Tuple[float, float], 
                                     p3: Tuple[float, float]) -> float:
    """
    세 점으로 이루어진 각도를 계산합니다.
    p2가 꼭짓점이 되는 각도 (p1-p2-p3)
    
    Args:
        p1, p2, p3: (x, y) 좌표 튜플
    
    Returns:
        각도 (degrees)
    """
    # 벡터 계산
    v1 = (p1[0] - p2[0], p1[1] - p2[1])  # p2 -> p1
    v2 = (p3[0] - p2[0], p3[1] - p2[1])  # p2 -> p3
    
    # 벡터의 크기
    len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # 영벡터 방지
    if len_v1 == 0 or len_v2 == 0:
        return 0.0
    
    # 코사인 값 계산
    cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
    
    # 부동소수점 오차 보정
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # 라디안을 도로 변환
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_line_angle(p1: Tuple[float, float], 
                        p2: Tuple[float, float]) -> float:
    """
    두 점을 잇는 직선의 수평선에 대한 각도를 계산합니다.
    
    Returns:
        각도 (degrees, -180 ~ 180)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dx == 0 and dy == 0:
        return 0.0
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_angle_between_lines(line1_p1: Tuple[float, float], 
                                 line1_p2: Tuple[float, float],
                                 line2_p1: Tuple[float, float], 
                                 line2_p2: Tuple[float, float]) -> float:
    """
    두 직선 사이의 각도를 계산합니다.
    
    Returns:
        각도 (degrees, 0 ~ 180)
    """
    angle1 = calculate_line_angle(line1_p1, line1_p2)
    angle2 = calculate_line_angle(line2_p1, line2_p2)
    
    diff = abs(angle1 - angle2)
    
    # 0~180도 범위로 정규화
    if diff > 180:
        diff = 360 - diff
    
    return diff

def calculate_sna(landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    SNA 각도를 계산합니다.
    SNA = Sella-Nasion-A point 각도
    """
    try:
        s_point = landmarks["S"]
        n_point = landmarks["N"] 
        a_point = landmarks["A"]
        
        return calculate_angle_from_three_points(s_point, n_point, a_point)
    except KeyError as e:
        raise ValueError(f"SNA 계산에 필요한 랜드마크가 없습니다: {e}")

def calculate_snb(landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    SNB 각도를 계산합니다.
    SNB = Sella-Nasion-B point 각도
    """
    try:
        s_point = landmarks["S"]
        n_point = landmarks["N"]
        b_point = landmarks["B"]
        
        return calculate_angle_from_three_points(s_point, n_point, b_point)
    except KeyError as e:
        raise ValueError(f"SNB 계산에 필요한 랜드마크가 없습니다: {e}")

def calculate_anb(sna: float, snb: float) -> float:
    """
    ANB 각도를 계산합니다.
    ANB = SNA - SNB
    """
    return sna - snb

def calculate_fma(landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    FMA 각도를 계산합니다.
    FMA = Frankfort Horizontal plane과 Mandibular plane 사이의 각도
    
    - Frankfort Horizontal (FH): Porion(Po) - Orbitale(Or)
    - Mandibular plane: Gonion(Go) - Menton(Me)
    """
    try:
        # Frankfort Horizontal plane
        po_point = landmarks["Po"]
        or_point = landmarks["Or"]
        
        # Mandibular plane  
        go_point = landmarks["Go"]
        me_point = landmarks["Me"]
        
        return calculate_angle_between_lines(po_point, or_point, go_point, me_point)
    except KeyError as e:
        raise ValueError(f"FMA 계산에 필요한 랜드마크가 없습니다: {e}")

def assess_metric_status(value: float, normal_range: Tuple[float, float]) -> str:
    """
    측정값이 정상 범위에 있는지 판정합니다.
    
    Returns:
        "normal", "low", "high" 중 하나
    """
    min_val, max_val = normal_range
    
    if value < min_val:
        return "low"
    elif value > max_val:
        return "high"
    else:
        return "normal"

def compute_all_metrics(landmarks: Dict[str, Tuple[float, float]], 
                       config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    모든 임상 지표를 계산하고 정상 범위와 비교합니다.
    
    Args:
        landmarks: 랜드마크 좌표 딕셔너리 {"name": (x, y), ...}
        config_path: 정상 범위 설정 파일 경로
    
    Returns:
        계산된 지표와 상태 정보
    """
    # 정상 범위 로드
    if config_path is None:
        config_path = "data/clinical_standards/normal_ranges.json"
    
    normal_ranges = load_normal_ranges(config_path)
    
    try:
        # 기본 각도 계산
        sna = calculate_sna(landmarks)
        snb = calculate_snb(landmarks) 
        anb = calculate_anb(sna, snb)
        fma = calculate_fma(landmarks)
        
        # 결과 구성
        results = {}
        
        for metric_name, value in [("SNA", sna), ("SNB", snb), ("ANB", anb), ("FMA", fma)]:
            metric_config = normal_ranges["metrics"].get(metric_name, {})
            normal_range = metric_config.get("normal_range", [0, 0])
            unit = metric_config.get("unit", "degrees")
            
            status = assess_metric_status(value, normal_range)
            
            results[metric_name] = {
                "value": round(value, 2),
                "unit": unit,
                "normal_range": normal_range,
                "status": status,
                "description": metric_config.get("description", ""),
                "clinical_significance": metric_config.get("clinical_significance", "")
            }
        
        return results
        
    except Exception as e:
        raise ValueError(f"임상 지표 계산 중 오류가 발생했습니다: {e}")

def validate_landmarks(landmarks: Dict[str, Tuple[float, float]]) -> bool:
    """
    필수 랜드마크가 모두 있는지 확인합니다.
    """
    required_landmarks = ["S", "N", "A", "B", "Po", "Or", "Go", "Me"]
    
    for landmark in required_landmarks:
        if landmark not in landmarks:
            raise ValueError(f"필수 랜드마크가 없습니다: {landmark}")
        
        x, y = landmarks[landmark]
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise ValueError(f"랜드마크 좌표가 유효하지 않습니다: {landmark} = {landmarks[landmark]}")
    
    return True

# 편의 함수
def compute_all(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    메인 계산 함수 (외부에서 호출용)
    """
    validate_landmarks(landmarks)
    return compute_all_metrics(landmarks)

# 테스트용 함수
def test_with_demo_data():
    """
    데모 데이터로 테스트합니다.
    """
    # 테스트용 랜드마크 (정규화된 좌표를 절대 좌표로 변환)
    demo_landmarks = {
        "N": (468, 115),   # 0.585 * 800, 0.192 * 600
        "S": (340, 189),   # 0.425 * 800, 0.315 * 600  
        "A": (508, 291),   # 0.635 * 800, 0.485 * 600
        "B": (484, 375),   # 0.605 * 800, 0.625 * 600
        "Po": (276, 213),  # 0.345 * 800, 0.355 * 600
        "Or": (412, 191),  # 0.515 * 800, 0.318 * 600  
        "Go": (324, 363),  # 0.405 * 800, 0.605 * 600
        "Me": (484, 423),  # 0.605 * 800, 0.705 * 600
        # 나머지 랜드마크들 (계산에 직접 사용되지 않음)
        "Ar": (308, 267), "ANS": (516, 279), "PNS": (388, 285),
        "U1": (534, 317), "L1": (516, 351), "Ls": (580, 309),
        "Li": (556, 351), "Pog": (524, 399), "Pog'": (588, 417),
        "Gn": (500, 417), "Pn": (604, 273)
    }
    
    print("🧪 임상 지표 계산 테스트")
    print("=" * 50)
    
    try:
        results = compute_all(demo_landmarks)
        
        for metric_name, data in results.items():
            status_icon = {"normal": "✅", "low": "⬇️", "high": "⬆️"}.get(data["status"], "❓")
            print(f"{status_icon} {metric_name}: {data['value']}° (정상: {data['normal_range']}°)")
            
        print("\n📊 종합 판정:")
        anb_value = results["ANB"]["value"]
        if 0 <= anb_value <= 4:
            classification = "Class I (정상)"
        elif anb_value > 4:
            classification = "Class II (상악 과성장)"
        else:
            classification = "Class III (하악 과성장)"
            
        print(f"   분류: {classification}")
        print(f"   ANB: {anb_value}°")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return None

if __name__ == "__main__":
    test_with_demo_data()