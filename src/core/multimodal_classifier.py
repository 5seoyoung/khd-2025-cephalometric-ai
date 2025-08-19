# -*- coding: utf-8 -*-
"""
Enhanced Multimodal Classifier
메타데이터(연령, 성별)를 적극 활용하여 개인화된 분류를 수행합니다.

개선사항:
- 연령/성별별 정상 범위 차별화
- 메타데이터 기반 특성 엔지니어링 강화
- 개인화된 분류 임계값
- 동적 가중치 조정
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

# 연령/성별별 정상 범위 (한국인 기준)
AGE_SEX_NORMS = {
    "ANB": {
        "child": {"M": [1.0, 5.0], "F": [1.5, 5.5], "default": [1.0, 5.0]},      # 10-15세
        "young_adult": {"M": [0.5, 4.0], "F": [1.0, 4.5], "default": [0.5, 4.5]}, # 16-25세
        "adult": {"M": [0.0, 3.5], "F": [0.5, 4.0], "default": [0.0, 4.0]},       # 26-40세
        "middle_aged": {"M": [-0.5, 3.0], "F": [0.0, 3.5], "default": [-0.5, 3.5]} # 41세+
    },
    "SNA": {
        "child": {"M": [78, 86], "F": [79, 87], "default": [78, 87]},
        "young_adult": {"M": [79, 85], "F": [80, 86], "default": [79, 86]},
        "adult": {"M": [80, 84], "F": [81, 85], "default": [80, 85]},
        "middle_aged": {"M": [81, 83], "F": [82, 84], "default": [81, 84]}
    },
    "SNB": {
        "child": {"M": [75, 83], "F": [76, 84], "default": [75, 84]},
        "young_adult": {"M": [76, 82], "F": [77, 83], "default": [76, 83]},
        "adult": {"M": [78, 82], "F": [79, 83], "default": [78, 83]},
        "middle_aged": {"M": [79, 81], "F": [80, 82], "default": [79, 82]}
    },
    "FMA": {
        "child": {"M": [22, 32], "F": [23, 33], "default": [22, 33]},
        "young_adult": {"M": [23, 31], "F": [24, 32], "default": [23, 32]},
        "adult": {"M": [25, 30], "F": [26, 31], "default": [25, 31]},
        "middle_aged": {"M": [26, 29], "F": [27, 30], "default": [26, 30]}
    }
}

def get_age_group(age: int) -> str:
    """연령대 분류"""
    if age <= 15:
        return "child"
    elif age <= 25:
        return "young_adult"
    elif age <= 40:
        return "adult"
    else:
        return "middle_aged"

def get_personalized_normal_range(metric: str, age: int, sex: str) -> Tuple[float, float]:
    """개인화된 정상 범위 반환"""
    age_group = get_age_group(age)
    
    if metric in AGE_SEX_NORMS:
        norms = AGE_SEX_NORMS[metric][age_group]
        if sex in norms:
            return tuple(norms[sex])
        else:
            return tuple(norms["default"])
    
    # 기본값 (기존 범위)
    defaults = {
        "ANB": (0, 4),
        "SNA": (80, 84),
        "SNB": (78, 82),
        "FMA": (25, 30)
    }
    return defaults.get(metric, (0, 10))

def calculate_personalized_deviation(value: float, metric: str, age: int, sex: str) -> float:
    """개인화된 정상 범위 기준 이탈도 계산"""
    min_norm, max_norm = get_personalized_normal_range(metric, age, sex)
    
    if value < min_norm:
        return (min_norm - value) / (max_norm - min_norm)  # 음수 이탈
    elif value > max_norm:
        return (value - max_norm) / (max_norm - min_norm)  # 양수 이탈
    else:
        return 0.0  # 정상 범위 내

def enhanced_rule_based_classification(anb_value: float, age: int, sex: str) -> Tuple[int, float]:
    """
    개인화된 규칙 기반 분류
    
    Returns:
        (predicted_class, confidence)
    """
    # 개인화된 정상 범위 가져오기
    min_norm, max_norm = get_personalized_normal_range("ANB", age, sex)
    
    # 개인화된 임계값 계산
    class_2_threshold = max_norm + 0.5  # 정상 상한 + 여유
    class_3_threshold = min_norm - 0.5  # 정상 하한 - 여유
    
    # 분류
    if anb_value > class_2_threshold:
        predicted_class = 1  # Class II
        # 임계값에서 멀수록 높은 신뢰도
        distance = anb_value - class_2_threshold
        confidence = min(0.95, 0.6 + distance * 0.1)
    elif anb_value < class_3_threshold:
        predicted_class = 2  # Class III
        distance = class_3_threshold - anb_value
        confidence = min(0.95, 0.6 + distance * 0.1)
    else:
        predicted_class = 0  # Class I
        # 정상 범위 중심에 가까울수록 높은 신뢰도
        center = (min_norm + max_norm) / 2
        distance_from_center = abs(anb_value - center)
        max_distance = (max_norm - min_norm) / 2
        confidence = 0.9 - (distance_from_center / max_distance) * 0.3
        confidence = max(0.6, confidence)
    
    return predicted_class, confidence

def extract_enhanced_features(metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """메타데이터를 적극 활용한 강화된 특성 추출"""
    features = {}
    
    # 기본 메타데이터
    age = meta.get("age", 25)
    sex = meta.get("sex", "U")
    features["age"] = age
    features["sex_encoded"] = {"M": 1, "F": 2, "U": 1.5}.get(sex, 1.5)
    
    # 연령대 분류
    age_group = get_age_group(age)
    features["age_group"] = {"child": 1, "young_adult": 2, "adult": 3, "middle_aged": 4}[age_group]
    
    # 임상 지표 특성
    sna = metrics.get("SNA", {}).get("value", 82)
    snb = metrics.get("SNB", {}).get("value", 80)
    anb = metrics.get("ANB", {}).get("value", 2)
    fma = metrics.get("FMA", {}).get("value", 27)
    
    features.update({
        "SNA": sna, "SNB": snb, "ANB": anb, "FMA": fma
    })
    
    # 개인화된 이탈도 계산
    for metric in ["ANB", "SNA", "SNB", "FMA"]:
        if metric in metrics:
            value = metrics[metric]["value"]
            deviation = calculate_personalized_deviation(value, metric, age, sex)
            features[f"{metric}_personalized_deviation"] = deviation
    
    # 연령-성별 상호작용 특성
    features["age_sex_interaction"] = age * features["sex_encoded"]
    
    # 성장 관련 특성 (청소년기 고려)
    if age <= 18:
        features["growth_stage"] = 1  # 성장기
        features["anb_growth_adjusted"] = anb + (18 - age) * 0.1  # 성장 보정
    else:
        features["growth_stage"] = 0  # 성장 완료
        features["anb_growth_adjusted"] = anb
    
    # 복합 지표
    features["sagittal_discrepancy"] = abs(anb - 2.0)  # 시상면 불조화도
    features["vertical_pattern"] = 1 if fma > 30 else 0  # 수직 성장 패턴
    
    # 성별별 특화 특성
    if sex == "F":
        # 여성: 일반적으로 약간 더 높은 ANB 허용
        features["sex_adjusted_anb"] = anb - 0.5
    elif sex == "M":
        # 남성: 일반적으로 약간 더 낮은 ANB 기대
        features["sex_adjusted_anb"] = anb + 0.5
    else:
        features["sex_adjusted_anb"] = anb
    
    return features

def enhanced_ml_simulation(features: Dict[str, Any], seed: int = 42) -> np.ndarray:
    """메타데이터를 적극 활용한 머신러닝 시뮬레이션"""
    # 특성 기반 시드 생성
    feature_hash = abs(hash(str(sorted(features.items())))) % 10000
    combined_seed = (seed + feature_hash) % (2**31)
    rng = np.random.RandomState(combined_seed)
    
    # 기본 로짓
    logits = rng.normal(0, 0.8, 3)
    
    # 메타데이터 기반 바이어스 강화
    age = features.get("age", 25)
    sex_encoded = features.get("sex_encoded", 1.5)
    anb = features.get("ANB", 2)
    
    # 개인화된 이탈도 활용
    anb_deviation = features.get("ANB_personalized_deviation", 0)
    
    # 강화된 도메인 지식 적용
    if anb_deviation > 0.5:  # 개인화된 정상 범위 초과
        if anb > 4:
            logits[1] += 1.5 + anb_deviation  # Class II 강화
        else:
            logits[2] += 1.5 + anb_deviation  # Class III 강화
    elif anb_deviation < 0.1:  # 정상 범위 내
        logits[0] += 1.0  # Class I 강화
    
    # 연령대별 조정
    age_group = features.get("age_group", 3)
    if age_group == 1:  # 어린이
        # 성장으로 인한 변동성 고려
        logits += rng.normal(0, 0.3, 3)
    elif age_group == 4:  # 중년
        # 안정적인 패턴
        logits[0] += 0.3  # Class I 경향
    
    # 성별별 조정
    if sex_encoded == 2:  # 여성
        logits[1] += 0.2  # Class II 약간 선호 (통계적 경향)
    elif sex_encoded == 1:  # 남성  
        logits[2] += 0.2  # Class III 약간 선호
    
    # 성장 단계 고려
    if features.get("growth_stage", 0) == 1:
        # 성장기에는 불안정성 증가
        logits += rng.normal(0, 0.2, 3)
    
    # 소프트맥스 변환
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities

def calculate_dynamic_weights(features: Dict[str, Any]) -> Tuple[float, float]:
    """메타데이터 기반 동적 가중치 계산"""
    age = features.get("age", 25)
    anb_deviation = features.get("ANB_personalized_deviation", 0)
    
    # 기본 가중치
    rule_weight = 0.6
    model_weight = 0.4
    
    # 이탈도가 클수록 규칙 가중치 증가
    if anb_deviation > 1.0:
        rule_weight = 0.8
        model_weight = 0.2
    elif anb_deviation > 0.5:
        rule_weight = 0.7
        model_weight = 0.3
    
    # 어린이는 모델 가중치 증가 (성장 변동성)
    if age <= 15:
        rule_weight -= 0.1
        model_weight += 0.1
    
    # 정규화
    total = rule_weight + model_weight
    return rule_weight / total, model_weight / total

class EnhancedDemoClassifier:
    """메타데이터를 적극 활용하는 향상된 데모 분류기"""
    
    def __init__(self, seed: int = 42):
        """초기화"""
        self.seed = seed
        print(f"✅ EnhancedDemoClassifier 초기화 완료 (seed={seed})")
    
    def predict(self, 
                metrics: Dict[str, Any], 
                meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """개인화된 부정교합 분류를 수행합니다."""
        if meta is None:
            meta = {"age": 25, "sex": "U"}
        
        # 강화된 특성 추출
        features = extract_enhanced_features(metrics, meta)
        
        age = features["age"]
        sex = meta.get("sex", "U")
        anb_value = features["ANB"]
        
        # 1. 개인화된 규칙 기반 분류
        rule_class, rule_confidence = enhanced_rule_based_classification(anb_value, age, sex)
        
        # 2. 강화된 ML 시뮬레이션
        model_probs = enhanced_ml_simulation(features, self.seed)
        model_class = int(np.argmax(model_probs))
        model_confidence = float(np.max(model_probs))
        
        # 3. 동적 가중치 계산
        rule_weight, model_weight = calculate_dynamic_weights(features)
        
        # 4. 최종 확률 계산
        final_probs = np.zeros(3)
        final_probs[rule_class] += rule_weight * rule_confidence
        final_probs += model_weight * model_probs
        
        # 정규화
        final_probs = final_probs / np.sum(final_probs)
        
        # 5. 최종 예측
        final_class = int(np.argmax(final_probs))
        final_confidence = float(np.max(final_probs))
        
        # 6. 개인화된 해석 생성
        personalized_basis = self._generate_personalized_basis(
            anb_value, age, sex, final_class, features
        )
        
        # 7. 결과 구성
        result = {
            "predicted_class": final_class,
            "predicted_label": CLASS_LABELS[final_class],
            "confidence": final_confidence,
            "probabilities": {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(final_probs)
            },
            "anb_value": anb_value,
            "personalized_analysis": {
                "age_group": get_age_group(age),
                "sex": sex,
                "normal_range_anb": get_personalized_normal_range("ANB", age, sex),
                "anb_deviation": features.get("ANB_personalized_deviation", 0),
                "classification_basis": personalized_basis
            },
            "model_weights": {
                "rule_based": rule_weight,
                "ml_based": model_weight,
                "explanation": f"연령 {age}세, 성별 {sex} 기준 개인화된 가중치"
            },
            "components": {
                "rule_based": {
                    "class": rule_class,
                    "confidence": rule_confidence,
                    "weight": rule_weight
                },
                "model_based": {
                    "class": model_class,
                    "confidence": model_confidence,
                    "weight": model_weight
                }
            }
        }
        
        return result
    
    def _generate_personalized_basis(self, anb_value: float, age: int, sex: str, 
                                   predicted_class: int, features: Dict[str, Any]) -> str:
        """개인화된 분류 근거 생성"""
        min_norm, max_norm = get_personalized_normal_range("ANB", age, sex)
        age_group_kr = {
            "child": "소아", "young_adult": "청년", 
            "adult": "성인", "middle_aged": "중년"
        }[get_age_group(age)]
        
        sex_kr = {"M": "남성", "F": "여성", "U": "미상"}[sex]
        
        basis_parts = []
        
        # 개인화된 ANB 분석
        if predicted_class == 1:  # Class II
            basis_parts.append(f"ANB {anb_value:.1f}° > {max_norm:.1f}° ({age_group_kr} {sex_kr} 정상상한)")
        elif predicted_class == 2:  # Class III
            basis_parts.append(f"ANB {anb_value:.1f}° < {min_norm:.1f}° ({age_group_kr} {sex_kr} 정상하한)")
        else:  # Class I
            basis_parts.append(f"ANB {anb_value:.1f}° ({age_group_kr} {sex_kr} 정상범위 {min_norm:.1f}-{max_norm:.1f}°)")
        
        # 추가 고려사항
        if age <= 15:
            basis_parts.append("성장기 변동성 고려")
        
        if features.get("growth_stage", 0) == 1:
            basis_parts.append("성장 보정 적용")
        
        return " | ".join(basis_parts)
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """분류기 정보 반환"""
        return {
            "classifier": "EnhancedDemoClassifier",
            "version": "2.0",
            "seed": self.seed,
            "features": [
                "개인화된 정상범위",
                "연령/성별별 차별화",
                "동적 가중치 조정",
                "성장기 보정",
                "메타데이터 융합"
            ],
            "personalization": {
                "age_groups": list(AGE_SEX_NORMS["ANB"].keys()),
                "sex_differentiation": True,
                "growth_adjustment": True
            }
        }

def test_enhanced_classifier():
    """향상된 분류기 테스트"""
    print("🧪 EnhancedDemoClassifier 테스트")
    print("=" * 60)
    
    try:
        classifier = EnhancedDemoClassifier(seed=42)
        
        # 다양한 연령/성별 테스트 케이스
        test_cases = [
            {
                "name": "소아 여아 (정상)",
                "metrics": {
                    "ANB": {"value": 3.5}, "SNA": {"value": 82.0},
                    "SNB": {"value": 78.5}, "FMA": {"value": 28.0}
                },
                "meta": {"age": 12, "sex": "F"}
            },
            {
                "name": "청년 남성 (Class II 경계)",
                "metrics": {
                    "ANB": {"value": 4.8}, "SNA": {"value": 84.0},
                    "SNB": {"value": 79.2}, "FMA": {"value": 26.0}
                },
                "meta": {"age": 22, "sex": "M"}
            },
            {
                "name": "중년 여성 (Class III)",
                "metrics": {
                    "ANB": {"value": -1.5}, "SNA": {"value": 80.0},
                    "SNB": {"value": 81.5}, "FMA": {"value": 29.0}
                },
                "meta": {"age": 45, "sex": "F"}
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 테스트 케이스 {i}: {case['name']}")
            
            result = classifier.predict(case["metrics"], case["meta"])
            
            print(f"   예측: {result['predicted_label']}")
            print(f"   신뢰도: {result['confidence']*100:.1f}%")
            
            # 개인화 분석
            analysis = result["personalized_analysis"]
            print(f"   개인화 정상범위: {analysis['normal_range_anb']}°")
            print(f"   이탈도: {analysis['anb_deviation']:.2f}")
            print(f"   근거: {analysis['classification_basis']}")
            
            # 가중치 정보
            weights = result["model_weights"]
            print(f"   가중치: 규칙 {weights['rule_based']*100:.0f}% / ML {weights['ml_based']*100:.0f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_classifier()