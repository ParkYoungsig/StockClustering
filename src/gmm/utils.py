"""GMM 파이프라인 공통 유틸.

- 전처리 스케일러, 메타데이터, 모델 등의 산출물(artifacts) 저장
"""

from pathlib import Path
import joblib


def save_artifacts(
    results_dir: Path,
    scaler,
    labels_map,
    final_k: int,
    features,
    means,
    model,
) -> None:
    """
    분석 결과물(Artifacts)을 파일로 저장합니다.

    [저장 항목]
    1. scaler.pkl: 전처리에 사용된 스케일러 객체
    2. metadata.pkl: 클러스터 라벨, 최종 K값, 사용된 Feature 목록 등 메타데이터
    3. gmm_latest_year.pkl: 학습된 GMM 모델 객체 (최신 연도 기준)

    Args:
        results_dir (Path): 결과 저장 최상위 경로
        scaler: 학습된 Scikit-learn Scaler 객체
        labels_map: 연도별 클러스터 라벨 딕셔너리
        final_k (int): 최종 선정된 클러스터 개수
        features (list): 사용된 Feature 컬럼명 리스트
        means (pd.DataFrame): 클러스터별 평균값 데이터프레임
        model: 학습된 GMM 모델 객체
    """
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(
        {
            "labels_per_year": labels_map,
            "final_k": final_k,
            "feature_columns": features,
            "cluster_means_latest": means,
        },
        artifacts_dir / "metadata.pkl",
    )

    if model is not None:
        joblib.dump(model, artifacts_dir / "gmm_latest_year.pkl")
