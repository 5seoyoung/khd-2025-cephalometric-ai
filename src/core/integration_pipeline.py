# -*- coding: utf-8 -*-
"""
Integration Pipeline
전체 cephalometric 분석 워크플로우를 통합 실행합니다.

워크플로우:
1) 이미지 입력 및 전처리
2) 랜드마크 추론 (demo_inference)
3) 임상 지표 계산 (clinical_metrics)
4) 부정교합 분류 (multimodal_classifier)
5) 결과 통합 및 반환
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

from PIL import Image

__all__ = ["CephalometricPipeline"]

# --------------------------------------------------------------------------------------
# 로깅 설정 (Streamlit/CLI 모두에서 보기 좋은 형식)
# --------------------------------------------------------------------------------------
logger = logging.getLogger("konyang.ceph.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------------
# 로컬 모듈 임포트 (패키지 실행/직접 실행 모두 지원)
# --------------------------------------------------------------------------------------
try:
    # 패키지 컨텍스트 (src를 패키지 루트로 인식)
    from .demo_inference import ImprovedDemoInference as DemoInference
    from .clinical_metrics import compute_all as compute_clinical_metrics
    from .multimodal_classifier import EnhancedDemoClassifier as DemoClassifier
except Exception:
    # 직접 실행/상대 경로 문제 대응
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from demo_inference import ImprovedDemoInference as DemoInference
    from clinical_metrics import compute_all as compute_clinical_metrics
    from multimodal_classifier import EnhancedDemoClassifier as DemoClassifier


class CephalometricPipeline:
    """
    측면두부규격방사선사진 분석 통합 파이프라인
    """

    # 구성 파일 기본 상대 경로
    _DEFAULT_CFG_DIR = Path("data") / "clinical_standards"
    _DEMO_LMK_FILE = "demo_landmarks.json"
    _MEAN_SHAPE_FILE = "mean_shape.json"

    def __init__(
        self,
        demo_mode: bool = True,
        seed: int = 42,
        rule_weight: float = 0.7,
        config_dir: Optional[Union[str, Path]] = None,
    ):
        """
        파이프라인 초기화 (경로/의존성 안전 버전)

        Args:
            demo_mode: 데모 모드 사용 여부
            seed: 난수 시드
            rule_weight: (향후 확장) 룰 기반 가중치
            config_dir: 임상 표준 설정 파일 디렉토리(미지정 시 자동 탐색)
        """
        self.demo_mode = demo_mode
        self.seed = seed
        self.rule_weight = rule_weight

        # ----------------------------- 설정 경로 확정 -----------------------------
        # 우선순위: 인자 > 환경변수(KONYANG_DATA_DIR) > 프로젝트 루트/data/clinical_standards
        if config_dir is not None:
            cfg_dir = Path(config_dir)
        else:
            env_dir = os.environ.get("KONYANG_DATA_DIR")
            if env_dir:
                cfg_dir = Path(env_dir)
            else:
                # integration_pipeline.py 위치: src/core/ -> 프로젝트 루트는 parent.parent
                project_root = Path(__file__).resolve().parent.parent.parent
                cfg_dir = project_root / self._DEFAULT_CFG_DIR

        self.config_dir: Path = cfg_dir
        self.demo_lmk_path = self.config_dir / self._DEMO_LMK_FILE
        self.mean_shape_path = self.config_dir / self._MEAN_SHAPE_FILE

        # 경로 존재 여부를 미리 로그로 알림(없어도 DemoInference가 자체 fallback을 가질 수 있음)
        if not self.config_dir.exists():
            logger.warning(f"임상 표준 디렉토리가 존재하지 않습니다: {self.config_dir}")
        if not self.demo_lmk_path.exists():
            logger.warning(f"데모 랜드마크 설정이 없습니다: {self.demo_lmk_path}")
        if not self.mean_shape_path.exists():
            logger.warning(f"평균 형태 설정이 없습니다: {self.mean_shape_path}")

        # ----------------------------- 컴포넌트 초기화 -----------------------------
        if demo_mode:
            try:
                self.inference_engine = DemoInference(
                    demo_config_path=str(self.demo_lmk_path),
                    mean_shape_path=str(self.mean_shape_path),
                    seed=seed,
                )
            except Exception as e:
                # 추론 엔진 초기화 실패 시, 파이프라인은 살아있되 run()에서 에러 리턴
                logger.exception("DemoInference 초기화 실패")
                self.inference_engine = None
                self._init_error = ("InferenceInitError", str(e))
            else:
                self._init_error = None

            try:
                self.classifier = DemoClassifier(seed=seed)
            except Exception as e:
                logger.exception("DemoClassifier 초기화 실패")
                self.classifier = None
                self._clf_init_error = ("ClassifierInitError", str(e))
            else:
                self._clf_init_error = None
        else:
            raise NotImplementedError("연구 모드는 안전 구역에서만 사용 가능합니다.")

        # ----------------------------- 실행 통계 -----------------------------
        self.stats: Dict[str, Any] = {
            "total_runs": 0,
            "last_run_time": None,
            "average_processing_time": 0.0,
        }

        logger.info(
            f"✅ CephalometricPipeline 초기화 완료 "
            f"(demo_mode={demo_mode}, cfg='{self.config_dir}')"
        )

    # ----------------------------------------------------------------------------------
    # 내부 유틸
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _ensure_pil_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """문자열 경로 또는 PIL.Image를 일관된 PIL.Image로 변환."""
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, os.PathLike)):
            path = str(image_input)
            if not os.path.exists(path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
            img = Image.open(path)
        else:
            raise ValueError("지원하지 않는 이미지 입력 형식입니다 (str 경로 또는 PIL.Image 필요)")

        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < 100 or h < 100:
            raise ValueError(f"이미지가 너무 작습니다: {w}x{h} (최소 100x100)")
        return img

    def _update_stats(self, processing_time: float) -> None:
        """실행 통계를 이동평균으로 갱신."""
        self.stats["total_runs"] += 1
        self.stats["last_run_time"] = processing_time
        if self.stats["total_runs"] == 1:
            self.stats["average_processing_time"] = processing_time
        else:
            alpha = 0.1
            self.stats["average_processing_time"] = (
                alpha * processing_time + (1 - alpha) * self.stats["average_processing_time"]
            )

    @staticmethod
    def _summarize_quality(
        landmarks: Dict[str, Tuple[float, float]],
        clinical_metrics: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """결과 품질을 평가하고 요약."""
        quality_scores: Dict[str, float] = {}
        warnings: List[str] = []

        # 1) 랜드마크 품질
        lmk_score = 1.0
        if len(landmarks) < 19:
            lmk_score -= 0.1 * (19 - len(landmarks))
            warnings.append(f"일부 랜드마크 누락 ({len(landmarks)}/19)")
        quality_scores["landmarks"] = max(0.0, min(1.0, lmk_score))

        # 2) 임상 지표 품질(정상/이상 비율)
        m_score = 1.0
        abnormal = sum(1 for v in clinical_metrics.values() if v.get("status") != "normal")
        if abnormal >= 3:
            m_score -= 0.2
            warnings.append(f"다수 지표 이상 ({abnormal}개)")
        quality_scores["metrics"] = max(0.0, min(1.0, m_score))

        # 3) 분류 신뢰도
        conf = float(classification.get("confidence", 0.0))
        if conf < 0.7:
            warnings.append(f"낮은 분류 신뢰도 ({conf * 100:.1f}%)")
        quality_scores["classification"] = max(0.0, min(1.0, conf))

        # 4) 종합 점수
        overall = (
            quality_scores["landmarks"] * 0.3
            + quality_scores["metrics"] * 0.3
            + quality_scores["classification"] * 0.4
        )
        overall = max(0.0, min(1.0, overall))

        def rec_text(s: float, warns: List[str]) -> str:
            if s >= 0.9:
                return "우수한 분석 결과입니다."
            if s >= 0.7:
                return "양호한 분석 결과입니다."
            if s >= 0.5:
                return "분석 결과를 신중히 검토하시기 바랍니다."
            return "분석 결과의 신뢰도가 낮습니다. 이미지 품질이나 랜드마크 위치를 확인해주세요."

        return {
            "overall_score": round(overall, 3),
            "component_scores": quality_scores,
            "warnings": warnings,
            "recommendation": rec_text(overall, warnings),
        }

    # ----------------------------------------------------------------------------------
    # 공개 API
    # ----------------------------------------------------------------------------------
    def preprocess_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """입력 이미지를 전처리하여 PIL.Image로 반환."""
        return self._ensure_pil_image(image_input)

    def run(
        self,
        image_input: Union[str, Image.Image],
        meta: Optional[Dict[str, Any]] = None,
        anchors: Optional[Dict[str, Tuple[float, float]]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        파이프라인 전체 실행.

        Returns:
            분석 결과 딕셔너리(오류 시 'error' 키 포함)
        """
        # 초기화 에러가 있었다면 즉시 리턴(앱은 계속 동작)
        if getattr(self, "_init_error", None):
            etype, emsg = self._init_error
            return {
                "success": False,
                "error": {"type": etype, "message": emsg, "stage": "init"},
                "demo_mode": self.demo_mode,
            }
        if getattr(self, "inference_engine", None) is None or getattr(self, "classifier", None) is None:
            return {
                "success": False,
                "error": {"type": "ComponentMissing", "message": "필수 컴포넌트가 초기화되지 않았습니다.", "stage": "init"},
                "demo_mode": self.demo_mode,
            }

        rid = run_id or str(uuid.uuid4())[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta = meta or {}

        try:
            start = time.perf_counter()

            # 1) 전처리
            t1 = time.perf_counter()
            image = self.preprocess_image(image_input)
            t2 = time.perf_counter()

            # 2) 랜드마크 추론
            landmarks, inference_mode = self.inference_engine.predict_landmarks(image, anchors=anchors)
            t3 = time.perf_counter()

            # 3) 임상 지표 계산
            clinical = compute_clinical_metrics(landmarks)
            t4 = time.perf_counter()

            # 4) 분류
            cls = self.classifier.predict(clinical, meta)
            t5 = time.perf_counter()

            total_s = t5 - start

            # 품질 요약
            quality = self._summarize_quality(landmarks, clinical, cls)

            result: Dict[str, Any] = {
                "run_id": rid,
                "timestamp": ts,
                "demo_mode": self.demo_mode,
                "seed": self.seed,
                "image_info": {"size": image.size, "mode": image.mode, "input_type": type(image_input).__name__},
                "meta": meta,
                "anchors_used": anchors is not None,
                "landmarks": {"count": len(landmarks), "inference_mode": inference_mode, "coordinates": landmarks},
                "clinical_metrics": clinical,
                "classification": cls,
                "performance": {
                    "total_time_ms": round(total_s * 1000, 2),
                    "preprocessing_ms": round((t2 - t1) * 1000, 2),
                    "inference_ms": round((t3 - t2) * 1000, 2),
                    "metrics_ms": round((t4 - t3) * 1000, 2),
                    "classification_ms": round((t5 - t4) * 1000, 2),
                },
                "quality": quality,
                "success": True,
            }

            self._update_stats(total_s)
            return result

        except Exception as e:
            logger.exception("파이프라인 실행 중 예외 발생")
            return {
                "run_id": rid,
                "timestamp": ts,
                "demo_mode": self.demo_mode,
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 상태/버전/컴포넌트 메타를 반환."""
        return {
            "pipeline": "CephalometricPipeline",
            "version": "1.0.1",
            "demo_mode": self.demo_mode,
            "seed": self.seed,
            "config_dir": str(self.config_dir),
            "components": {
                "inference_engine": getattr(self.inference_engine, "get_inference_info", lambda: {"name": "unknown"})(),
                "classifier": getattr(self.classifier, "get_classifier_info", lambda: {"name": "unknown"})(),
            },
            "statistics": dict(self.stats),
        }

    def run_batch(
        self,
        image_list: List[Union[str, Image.Image]],
        meta_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """여러 이미지를 순차 처리."""
        if meta_list is None:
            meta_list = [{} for _ in image_list]
        if len(meta_list) != len(image_list):
            raise ValueError("이미지 개수와 메타데이터 개수가 일치해야 합니다.")

        results: List[Dict[str, Any]] = []
        batch_start = time.perf_counter()

        logger.info(f"🔄 배치 처리 시작: {len(image_list)}개 이미지")
        for i, (img, meta) in enumerate(zip(image_list, meta_list), start=1):
            rid = f"batch_{i:03d}"
            try:
                res = self.run(img, meta=meta, run_id=rid)
                results.append(res)
                if res.get("success"):
                    ms = res["performance"]["total_time_ms"]
                    logger.info(f"   ✅ {i}/{len(image_list)} 완료 ({ms:.1f}ms)")
                else:
                    logger.warning(f"   ⚠️ {i}/{len(image_list)} 실패: {res.get('error', {}).get('message')}")
            except Exception as e:
                results.append({"run_id": rid, "success": False, "error": {"type": type(e).__name__, "message": str(e)}})
                logger.exception(f"   ❌ {i}/{len(image_list)} 예외")

        logger.info(f"🏁 배치 처리 완료: {time.perf_counter() - batch_start:.2f}s")
        return results


# --------------------------------------------------------------------------------------
# 로컬 테스트용 진입점
# --------------------------------------------------------------------------------------
def test_integration_pipeline() -> bool:
    """
    통합 파이프라인 간단 테스트 (로컬 실행 전용)
    """
    print("🧪 CephalometricPipeline 테스트")
    print("=" * 60)

    try:
        pipeline = CephalometricPipeline(demo_mode=True, seed=42)
        info = pipeline.get_pipeline_info()
        print("📋 파이프라인 정보:", json.dumps(info, ensure_ascii=False, indent=2))

        # 테스트용 더미 이미지
        img = Image.new("RGB", (800, 600), color="#DDDDDD")

        print("\n🚀 기본 실행 테스트:")
        meta = {"age": 25, "sex": "F", "patient_id": "TEST001"}
        result = pipeline.run(img, meta=meta)

        if not result.get("success", False):
            print(f"   ❌ 실행 실패: {result.get('error')}")
            return False

        print(f"   ✅ 실행 성공 (ID: {result['run_id']})")
        print(f"   총 처리 시간: {result['performance']['total_time_ms']:.1f}ms")
        print(f"   추론 모드: {result['landmarks']['inference_mode']}")
        print(f"   분류 결과: {result['classification'].get('predicted_label')}")
        print(f"   신뢰도: {result['classification'].get('confidence', 0.0) * 100:.1f}%")
        print(f"   품질 점수: {result['quality']['overall_score']:.3f}")

        print("\n🔧 앵커 포인트 테스트:")
        anchors = {"Or": (400.0, 200.0), "Po": (300.0, 210.0)}
        res2 = pipeline.run(img, meta=meta, anchors=anchors)
        print(f"   앵커 사용: {res2.get('anchors_used')} / 모드: {res2['landmarks']['inference_mode']}")

        print("\n📊 성능 세부:")
        for k, v in result["performance"].items():
            if k.endswith("_ms"):
                print(f"   {k:>18}: {v:>7.1f} ms")

        print("\n🔍 품질 평가:")
        q = result["quality"]
        print(f"   전체 점수: {q['overall_score']:.3f}")
        print(f"   권장사항: {q['recommendation']}")
        if q["warnings"]:
            print(f"   경고사항: {', '.join(q['warnings'])}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_integration_pipeline()