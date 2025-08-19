# -*- coding: utf-8 -*-
"""
Multimodal Classifier
임상 지표와 메타데이터를 결합하여 골격성 부정교합을 분류합니다.

주요 기능:
- Rule-based 분류 (ANB 기준)
- Seeded XGBoost 시뮬레이션 
- 신뢰도 계산 (결정경계와의 거리 기반)
- 분류 근거 제공
"""

import numpy as np
import json
from typing import Dict, Any, Tuple, Optional
import math

# 부정교합 분류 상수
CLASS_LABELS = {
    0: "Class I",
    1: "Class II", 
    2: "Class III"
}

CLASS_DESCRIPTIONS = {
    0: "골격적으로 정상",
    1: "골격적으로 상악 과성장 또는 하악 열성장",
    2: "골격적으로 하악 과성장 또는 상악 열성장"
}

# ANB 기준 임계값
ANB_THRESHOLDS = {
    "class_1_range": [0, 4],      # Class I: 0° ≤ ANB ≤ 4°
    "class_2_threshold": 4,       # Class II: ANB > 4°
    "class_3_threshold": 0        # Class III: ANB < 0°
}

def rule_based_classification(anb_value: float) -> int:
    """
    ANB 각도 기반 규칙 분류
    
    Args:
        anb_value: ANB 각도 (degrees)
    
    Returns:
        클래스 번호 (0: Class I, 1: Class II, 2: Class III)
    """
    if anb_value > ANB_THRESHOLDS["class_2_threshold"]:
        return 1  # Class II
    elif anb_value < ANB_THRESHOLDS["class_3_threshold"]:
        return 2  # Class III
    else:
        return 0  # Class I

def calculate_confidence_from_distance(anb_value: float, 
                                     predicted_class: int, 
                                     sigma: float = 2.0) -> float:
    """
    결정경계로부터의 거리를 기반으로 신뢰도를 계산합니다.
    
    Args:
        anb_value: ANB 각도
        predicted_class: 예측된 클래스
        sigma: 신뢰도 계산용 스케일링 파라미터
    
    Returns:
        신뢰도 (0.0 ~ 1.0)
    """
    if predicted_class == 1:  # Class II
        # ANB > 4인 경우, 4로부터의 거리
        distance = max(0, anb_value - ANB_THRESHOLDS["class_2_threshold"])
    elif predicted_class == 2:  # Class III  
        # ANB < 0인 경우, 0으로부터의 거리
        distance = max(0, ANB_THRESHOLDS["class_3_threshold"] - anb_value)
    else:  # Class I
        # 0~4 범위 내에서 중심(2)으로부터의 거리
        center = 2.0
        distance_from_center = abs(anb_value - center)
        # Class I의 경우 중심에 가까울수록 높은 신뢰도
        return 1.0 - min(1.0, distance_from_center / 2.0)
    
    # 시그모이드 함수로 신뢰도 계산
    confidence = 1.0 - math.exp(-distance / sigma)
    return min(0.95, max(0.5, confidence))  # 0.5~0.95 범위로 제한

def seeded_xgboost_simulation(features: Dict[str, Any], seed: int = 42) -> np.ndarray:
    """
    XGBoost 시뮬레이션 (결정론적 난수 기반)
    
    실제 XGBoost 대신 특성값 기반 해시와 시드를 조합하여
    일관된 확률 분포를 생성합니다.
    
    Args:
        features: 입력 특성들
        seed: 난수 시드
    
    Returns:
        3개 클래스에 대한 확률 분포
    """
    # 특성값들을 문자열로 변환하여 해시 생성
    feature_str = ""
    for key in sorted(features.keys()):
        if isinstance(features[key], (int, float)):
            feature_str += f"{key}:{features[key]:.2f};"
        else:
            feature_str += f"{key}:{features[key]};"
    
    # 특성 기반 해시를 시드와 조합
    feature_hash = abs(hash(feature_str)) % 10000
    combined_seed = (seed + feature_hash) % (2**31)
    
    # 시드 고정 난수 생성기
    rng = np.random.RandomState(combined_seed)
    
    # 3개 클래스에 대한 로짓 생성
    logits = rng.normal(0, 1, 3)
    
    # ANB 값에 따른 바이어스 추가 (약간의 도메인 지식 반영)
    anb = features.get("ANB", 2.0)
    if anb > 4:
        logits[1] += 0.5  # Class II 선호
    elif anb < 0:
        logits[2] += 0.5  # Class III 선호
    else:
        logits[0] += 0.3  # Class I 선호
    
    # 소프트맥스로 확률 변환
    exp_logits = np.exp(logits - np.max(logits))  # 수치 안정성
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities

def extract_features_from_metrics(metrics: Dict[str, Any], 
                                meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    임상 지표와 메타데이터로부터 분류용 특성을 추출합니다.
    """
    features = {}
    
    # 임상 지표 특성
    for metric_name in ["SNA", "SNB", "ANB", "FMA"]:
        if metric_name in metrics:
            features[metric_name] = metrics[metric_name]["value"]
            features[f"{metric_name}_status"] = metrics[metric_name]["status"]
    
    # 메타데이터 특성
    features["age"] = meta.get("age", 25)
    features["sex"] = meta.get("sex", "U")
    
    # 파생 특성
    sna = features.get("SNA", 82)
    snb = features.get("SNB", 80)
    features["SNA_SNB_diff"] = sna - snb  # ANB와 동일하지만 독립 계산
    
    # 정상 범위 이탈 정도
    anb = features.get("ANB", 2)
    if anb > 4:
        features["ANB_deviation"] = anb - 4
    elif anb < 0:
        features["ANB_deviation"] = abs(anb)
    else:
        features["ANB_deviation"] = 0
    
    return features

class DemoClassifier:
    """
    데모용 멀티모달 분류기
    """
    
    def __init__(self, seed: int = 42, rule_weight: float = 0.7):
        """
        초기화
        
        Args:
            seed: 난수 시드
            rule_weight: 규칙 기반 분류의 가중치 (0~1)
        """
        self.seed = seed
        self.rule_weight = rule_weight
        self.model_weight = 1.0 - rule_weight
        
        print(f"✅ DemoClassifier 초기화 완료 (seed={seed}, rule_weight={rule_weight})")
    
    def predict(self, 
                metrics: Dict[str, Any], 
                meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        부정교합 분류를 수행합니다.
        
        Args:
            metrics: clinical_metrics.compute_all()의 결과
            meta: 메타데이터 (나이, 성별 등)
        
        Returns:
            분류 결과 딕셔너리
        """
        if meta is None:
            meta = {}
        
        # ANB 값 추출
        anb_value = metrics["ANB"]["value"]
        
        # 1. 규칙 기반 분류
        rule_class = rule_based_classification(anb_value)
        rule_confidence = calculate_confidence_from_distance(anb_value, rule_class)
        
        # 2. 특성 추출
        features = extract_features_from_metrics(metrics, meta)
        
        # 3. 시뮬레이션 모델 예측
        model_probs = seeded_xgboost_simulation(features, self.seed)
        model_class = int(np.argmax(model_probs))
        model_confidence = float(np.max(model_probs))
        
        # 4. 가중 평균으로 최종 확률 계산
        final_probs = np.zeros(3)
        final_probs[rule_class] += self.rule_weight
        final_probs += self.model_weight * model_probs
        
        # 정규화
        final_probs = final_probs / np.sum(final_probs)
        
        # 5. 최종 예측
        final_class = int(np.argmax(final_probs))
        final_confidence = float(np.max(final_probs))
        
        # 6. 결과 구성
        result = {
            "predicted_class": final_class,
            "predicted_label": CLASS_LABELS[final_class],
            "confidence": final_confidence,
            "probabilities": {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(final_probs)
            },
            "anb_value": anb_value,
            "classification_basis": self._get_classification_basis(anb_value, final_class),
            "components": {
                "rule_based": {
                    "class": rule_class,
                    "confidence": rule_confidence,
                    "weight": self.rule_weight
                },
                "model_based": {
                    "class": model_class, 
                    "confidence": model_confidence,
                    "weight": self.model_weight
                }
            }
        }
        
        return result
    
    def _get_classification_basis(self, anb_value: float, predicted_class: int) -> str:
        """
        분류 근거를 텍스트로 반환합니다.
        """
        basis_parts = []
        
        # ANB 기반 근거
        if predicted_class == 1:  # Class II
            basis_parts.append(f"ANB {anb_value:.1f}° > 4° (상악 과성장 지시)")
        elif predicted_class == 2:  # Class III
            basis_parts.append(f"ANB {anb_value:.1f}° < 0° (하악 과성장 지시)")
        else:  # Class I
            basis_parts.append(f"ANB {anb_value:.1f}° (정상 범위 0-4°)")
        
        # 신뢰도 정보
        basis_parts.append(f"규칙 기반 {self.rule_weight*100:.0f}% + 모델 기반 {self.model_weight*100:.0f}%")
        
        return " | ".join(basis_parts)
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        분류기 정보를 반환합니다.
        """
        return {
            "classifier": "DemoClassifier",
            "version": "1.0",
            "seed": self.seed,
            "rule_weight": self.rule_weight,
            "model_weight": self.model_weight,
            "classes": list(CLASS_LABELS.values()),
            "primary_feature": "ANB angle"
        }

def test_multimodal_classifier():
    """
    멀티모달 분류기 테스트
    """
    print("🧪 DemoClassifier 테스트")
    print("=" * 50)
    
    try:
        # 1. 분류기 초기화
        classifier = DemoClassifier(seed=42, rule_weight=0.7)
        
        # 2. 분류기 정보 출력
        info = classifier.get_classifier_info()
        print("📋 분류기 정보:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 3. 테스트 케이스들
        test_cases = [
            {
                "name": "Class I (정상)",
                "metrics": {
                    "SNA": {"value": 82.5, "status": "normal"},
                    "SNB": {"value": 80.2, "status": "normal"},
                    "ANB": {"value": 2.3, "status": "normal"},
                    "FMA": {"value": 27.8, "status": "normal"}
                },
                "meta": {"age": 25, "sex": "F"}
            },
            {
                "name": "Class II (상악 과성장)",
                "metrics": {
                    "SNA": {"value": 85.0, "status": "high"},
                    "SNB": {"value": 78.0, "status": "normal"},
                    "ANB": {"value": 7.0, "status": "high"},
                    "FMA": {"value": 28.0, "status": "normal"}
                },
                "meta": {"age": 30, "sex": "M"}
            },
            {
                "name": "Class III (하악 과성장)",
                "metrics": {
                    "SNA": {"value": 80.0, "status": "normal"},
                    "SNB": {"value": 85.0, "status": "high"},
                    "ANB": {"value": -5.0, "status": "low"},
                    "FMA": {"value": 26.0, "status": "normal"}
                },
                "meta": {"age": 28, "sex": "F"}
            }
        ]
        
        # 4. 각 테스트 케이스 실행
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 테스트 케이스 {i}: {case['name']}")
            
            result = classifier.predict(case["metrics"], case["meta"])
            
            print(f"   예측 결과: {result['predicted_label']}")
            print(f"   신뢰도: {result['confidence']*100:.1f}%")
            print(f"   ANB: {result['anb_value']:.1f}°")
            print(f"   근거: {result['classification_basis']}")
            
            # 확률 분포 출력
            print("   확률 분포:")
            for label, prob in result["probabilities"].items():
                print(f"     {label}: {prob*100:.1f}%")
        
        # 5. 재현성 테스트
        print("\n🔄 재현성 테스트:")
        result1 = classifier.predict(test_cases[0]["metrics"], test_cases[0]["meta"])
        result2 = classifier.predict(test_cases[0]["metrics"], test_cases[0]["meta"])
        
        confidence_diff = abs(result1["confidence"] - result2["confidence"])
        print(f"   신뢰도 차이: {confidence_diff:.6f} (0이면 완전 재현)")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multimodal_classifier()