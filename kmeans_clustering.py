from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------------------------------------
# sys.path 세팅 (중요)
# ------------------------------------------------------------
# 현재 구조:
#   <repo_root>/
#     main.py
#     kmeans_clustering.py   <- (이 파일)
#     src/
#       kmm/
#         kmm_ready.py
#         kmm_pipeline.py
#         kmm_align_clusters_mixed.py
#         kmm_pca_maps_aligned.py
#         config.py
#
# kmm_ready.py 내부가 "from kmm_pipeline import ..." 처럼 "패키지 접두어 없이" 임포트하고 있어서 :contentReference[oaicite:4]{index=4}
# 1) src 를 sys.path에 추가 (kmm 패키지 접근용)
# 2) src/kmm 를 sys.path에 추가 (kmm_ready의 'bare import' 호환용)
# 을 둘 다 해주는 게 가장 안전합니다.
# ------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_KMM = _SRC / "kmm"

if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if _KMM.exists() and str(_KMM) not in sys.path:
    sys.path.insert(0, str(_KMM))


from kmm_ready import (
    run_kmeans_clustering,  # type: ignore  :contentReference[oaicite:5]{index=5}
)

# 이제 kmm_ready / kmm.config 등이 정상 import 됩니다.
from kmm.config import get_config  # type: ignore


class Kmms:
    """
    main.py에서
        import kmeans_clustering
        kms = kmeans_clustering.Kmms()
        kms.run()
    으로 실행하기 위한 래퍼 클래스.
    """

    def __init__(self) -> None:
        self.result: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        cfg = get_config()
        self.result = run_kmeans_clustering(cfg)
        return self.result
        self.result = run_kmeans_clustering(cfg)
        return self.result
        self.result = run_kmeans_clustering(cfg)
        return self.result
        self.result = run_kmeans_clustering(cfg)
        return self.result
        self.result = run_kmeans_clustering(cfg)
        return self.result
        return self.result
        return self.result
        return self.result
        return self.result
