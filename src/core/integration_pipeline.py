# -*- coding: utf-8 -*-
"""
Integration Pipeline
전체 cephalometric 분석 워크플로우를 통합 실행합니다.

워크플로우:
1. 이미지 입력 및 전처리
2. 랜드마크 추론 (demo_inference)
3. 임상 지표 계산 (clinical_metrics)
4. 부정교합 분류 (multimodal_classifier)
5. 결과 통합 및 반환
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
from PIL import Image
import json

# 로컬 모듈 임포트
try:
    # 패키지 내에서 실행시
    from .demo_inference import DemoInference
    from .clinical_metrics import compute_all as compute_clinical_metrics
    from .multimodal_classifier import DemoClassifier
except ImportError:
    # 직접 실행시
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from demo_inference import DemoInference
    from clinical_metrics import compute_all as compute_clinical_metrics
    from multimodal_classifier import DemoClassifier

class CephalometricPipeline:
    """
    측면두부규격방사선사진 분석 통합 파이프라인
    """
    
    def __init__(self, 
                 demo_mode: bool = True,
                 seed: int = 42,
                 rule_weight: float = 0.7,
                 config_dir: str = "data/clinical_standards"):
        """
        파이프라인 초기화
        
        Args:
            demo_mode: 데모 모드 활성화 여부
            seed: 재현성을 위한 시드
            rule_weight: 분류기의 규칙 기반 가중치
            config_dir: 설정 파일 디렉토리
        """
        self.demo_mode = demo_mode
        self.seed = seed
        self.config_dir = config_dir
        
        # 컴포넌트 초기화
        if demo_mode:
            self.inference_engine = DemoInference(
                demo_config_path=os.path.join(config_dir, "demo_landmarks.json"),
                mean_shape_path=os.path.join(config_dir, "mean_shape.json"),
                seed=seed
            )
            self.classifier = DemoClassifier(seed=seed, rule_weight=rule_weight)
        else:
            raise NotImplementedError("연구 모드는 안심존에서만 사용 가능합니다")
        
        # 실행 통계
        self.stats = {
            "total_runs": 0,
            "last_run_time": None,
            "average_processing_time": 0.0
        }
        
        print(f"✅ CephalometricPipeline 초기화 완료 (demo_mode={demo_mode})")
    
    def preprocess_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        입력 이미지를 전처리합니다.
        
        Args:
            image_input: 이미지 파일 경로 또는 PIL Image 객체
        
        Returns:
            전처리된 PIL Image
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("지원하지 않는 이미지 입력 형식입니다")
        
        # RGB 변환 (필요시)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 크기 검증
        width, height = image.size
        if width < 100 or height < 100:
            raise ValueError(f"이미지가 너무 작습니다: {width}x{height}")
        
        return image
    
    def run(self, 
            image_input: Union[str, Image.Image],
            meta: Optional[Dict[str, Any]] = None,
            anchors: Optional[Dict[str, Tuple[float, float]]] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 분석 파이프라인을 실행합니다.
        
        Args:
            image_input: 입력 이미지
            meta: 환자 메타데이터 (나이, 성별 등)
            anchors: 수동 앵커 포인트 (Or, Po)
            run_id: 실행 ID (미제공시 자동 생성)
        
        Returns:
            분석 결과 딕셔너리
        """
        # 실행 ID 및 타이밍 설정
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        
        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if meta is None:
            meta = {}
        
        try:
            # 1단계: 이미지 전처리
            t1 = time.perf_counter()
            image = self.preprocess_image(image_input)
            t2 = time.perf_counter()
            
            # 2단계: 랜드마크 추론
            landmarks, inference_mode = self.inference_engine.predict_landmarks(
                image, anchors=anchors
            )
            t3 = time.perf_counter()
            
            # 3단계: 임상 지표 계산
            clinical_metrics = compute_clinical_metrics(landmarks)
            t4 = time.perf_counter()
            
            # 4단계: 부정교합 분류
            classification_result = self.classifier.predict(clinical_metrics, meta)
            t5 = time.perf_counter()
            
            # 총 처리 시간
            total_time = t5 - start_time
            
            # 결과 통합
            result = {
                # 메타정보
                "run_id": run_id,
                "timestamp": timestamp,
                "demo_mode": self.demo_mode,
                "seed": self.seed,
                
                # 입력 정보
                "image_info": {
                    "size": image.size,
                    "mode": image.mode,
                    "input_type": type(image_input).__name__
                },
                "meta": meta,
                "anchors_used": anchors is not None,
                
                # 처리 결과
                "landmarks": {
                    "count": len(landmarks),
                    "inference_mode": inference_mode,
                    "coordinates": landmarks
                },
                "clinical_metrics": clinical_metrics,
                "classification": classification_result,
                
                # 성능 지표
                "performance": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "preprocessing_ms": round((t2 - t1) * 1000, 2),
                    "inference_ms": round((t3 - t2) * 1000, 2),
                    "metrics_ms": round((t4 - t3) * 1000, 2),
                    "classification_ms": round((t5 - t4) * 1000, 2)
                },
                
                # 품질 지표
                "quality": self._assess_result_quality(landmarks, clinical_metrics, classification_result)
            }
            
            # 통계 업데이트
            self._update_stats(total_time)
            
            return result
            
        except Exception as e:
            # 에러 정보 포함한 결과 반환
            error_result = {
                "run_id": run_id,
                "timestamp": timestamp,
                "demo_mode": self.demo_mode,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "occurred_at": time.perf_counter() - start_time
                },
                "success": False
            }
            
            print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            return error_result
    
    def _assess_result_quality(self, 
                              landmarks: Dict[str, Tuple[float, float]],
                              clinical_metrics: Dict[str, Any],
                              classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 품질을 평가합니다.
        """
        quality_scores = {}
        warnings = []
        
        # 1. 랜드마크 품질 평가
        landmark_score = 1.0
        if len(landmarks) < 19:
            landmark_score -= 0.1 * (19 - len(landmarks))
            warnings.append(f"일부 랜드마크 누락 ({len(landmarks)}/19)")
        
        quality_scores["landmarks"] = max(0.0, landmark_score)
        
        # 2. 임상 지표 품질 평가
        metrics_score = 1.0
        abnormal_count = 0
        for metric_name, metric_data in clinical_metrics.items():
            if metric_data["status"] != "normal":
                abnormal_count += 1
        
        if abnormal_count >= 3:
            warnings.append(f"다수 지표 이상 ({abnormal_count}/4)")
            metrics_score -= 0.2
        
        quality_scores["metrics"] = max(0.0, metrics_score)
        
        # 3. 분류 신뢰도 평가
        classification_confidence = classification["confidence"]
        if classification_confidence < 0.7:
            warnings.append(f"낮은 분류 신뢰도 ({classification_confidence*100:.1f}%)")
        
        quality_scores["classification"] = classification_confidence
        
        # 4. 전체 품질 점수
        overall_score = (
            quality_scores["landmarks"] * 0.3 + 
            quality_scores["metrics"] * 0.3 + 
            quality_scores["classification"] * 0.4
        )
        
        return {
            "overall_score": round(overall_score, 3),
            "component_scores": quality_scores,
            "warnings": warnings,
            "recommendation": self._get_quality_recommendation(overall_score, warnings)
        }
    
    def _get_quality_recommendation(self, score: float, warnings: list[str]) -> str:
        """
        품질 점수에 따른 권장사항을 반환합니다.
        """
        if score >= 0.9:
            return "우수한 분석 결과입니다."
        elif score >= 0.7:
            return "양호한 분석 결과입니다."
        elif score >= 0.5:
            return "분석 결과를 신중히 검토하시기 바랍니다."
        else:
            return "분석 결과의 신뢰도가 낮습니다. 이미지 품질이나 랜드마크 위치를 확인해주세요."
    
    def _update_stats(self, processing_time: float):
        """
        실행 통계를 업데이트합니다.
        """
        self.stats["total_runs"] += 1
        self.stats["last_run_time"] = processing_time
        
        # 이동평균으로 평균 처리 시간 업데이트
        if self.stats["total_runs"] == 1:
            self.stats["average_processing_time"] = processing_time
        else:
            alpha = 0.1  # 가중치
            self.stats["average_processing_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_processing_time"]
            )
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        파이프라인 정보를 반환합니다.
        """
        return {
            "pipeline": "CephalometricPipeline",
            "version": "1.0",
            "demo_mode": self.demo_mode,
            "seed": self.seed,
            "components": {
                "inference_engine": self.inference_engine.get_inference_info(),
                "classifier": self.classifier.get_classifier_info()
            },
            "statistics": self.stats.copy()
        }
    
    def run_batch(self, 
                  image_list: list[Union[str, Image.Image]],
                  meta_list: Optional[list[Dict[str, Any]]] = None) -> list[Dict[str, Any]]:
        """
        배치 처리를 수행합니다.
        """
        if meta_list is None:
            meta_list = [{}] * len(image_list)
        
        if len(meta_list) != len(image_list):
            raise ValueError("이미지와 메타데이터 개수가 일치하지 않습니다")
        
        results = []
        batch_start = time.perf_counter()
        
        print(f"🔄 배치 처리 시작: {len(image_list)}개 이미지")
        
        for i, (image, meta) in enumerate(zip(image_list, meta_list)):
            try:
                result = self.run(image, meta, run_id=f"batch_{i+1:03d}")
                results.append(result)
                print(f"   ✅ {i+1}/{len(image_list)} 완료 ({result['performance']['total_time_ms']:.1f}ms)")
            except Exception as e:
                error_result = {"run_id": f"batch_{i+1:03d}", "error": str(e), "success": False}
                results.append(error_result)
                print(f"   ❌ {i+1}/{len(image_list)} 실패: {e}")
        
        batch_time = time.perf_counter() - batch_start
        print(f"🏁 배치 처리 완료: {batch_time:.2f}초")
        
        return results

def test_integration_pipeline():
    """
    통합 파이프라인 테스트
    """
    print("🧪 CephalometricPipeline 테스트")
    print("=" * 60)
    
    try:
        # 1. 파이프라인 초기화
        pipeline = CephalometricPipeline(demo_mode=True, seed=42)
        
        # 2. 파이프라인 정보 출력
        info = pipeline.get_pipeline_info()
        print("📋 파이프라인 정보:")
        print(f"   버전: {info['version']}")
        print(f"   데모 모드: {info['demo_mode']}")
        print(f"   시드: {info['seed']}")
        
        # 3. 테스트용 이미지 생성
        test_image = Image.new("RGB", (800, 600), color="lightgray")
        
        # 4. 기본 실행 테스트
        print("\n🚀 기본 실행 테스트:")
        meta = {"age": 25, "sex": "F", "patient_id": "TEST001"}
        
        result = pipeline.run(test_image, meta=meta)
        
        if "error" in result:
            print(f"   ❌ 실행 실패: {result['error']}")
            return False
        
        print(f"   ✅ 실행 성공 (ID: {result['run_id']})")
        print(f"   총 처리 시간: {result['performance']['total_time_ms']:.1f}ms")
        print(f"   추론 모드: {result['landmarks']['inference_mode']}")
        print(f"   분류 결과: {result['classification']['predicted_label']}")
        print(f"   신뢰도: {result['classification']['confidence']*100:.1f}%")
        print(f"   품질 점수: {result['quality']['overall_score']:.3f}")
        
        # 5. 앵커 포인트 테스트
        print("\n🔧 앵커 포인트 테스트:")
        anchors = {"Or": (400, 200), "Po": (300, 210)}
        result_with_anchors = pipeline.run(test_image, meta=meta, anchors=anchors)
        
        print(f"   앵커 사용: {result_with_anchors['anchors_used']}")
        print(f"   추론 모드: {result_with_anchors['landmarks']['inference_mode']}")
        
        # 6. 성능 비교
        print("\n📊 성능 분석:")
        perf = result["performance"]
        for stage, time_ms in perf.items():
            if stage.endswith("_ms"):
                stage_name = stage.replace("_ms", "").replace("_", " ").title()
                print(f"   {stage_name}: {time_ms:.1f}ms")
        
        # 7. 품질 평가
        print("\n🔍 품질 평가:")
        quality = result["quality"]
        print(f"   전체 점수: {quality['overall_score']:.3f}")
        print(f"   권장사항: {quality['recommendation']}")
        if quality["warnings"]:
            print(f"   경고사항: {', '.join(quality['warnings'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_integration_pipeline()