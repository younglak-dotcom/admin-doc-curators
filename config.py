# config.py

# ===== 데이터 경로 =====
# 행정문서-순화문 병렬 데이터 (source, target, domain 컬럼)
PARALLEL_DATA_PATH = "data/admin_parallel.csv"

# 큐레이션용 문서 데이터 (doc_id, text, domain 컬럼)
ADMIN_DOCS_PATH = "data/admin_docs.csv"

# ===== 베이스 모델 =====
# 제안서에 쓴 순화/요약 모델 이름 반영
SIMPLIFIER_BASE_MODEL = "paust/pko-t5-base"
SUMMARIZER_BASE_MODEL = "psyche/KoT5-summarization"

# 파인튜닝 후 저장 위치
SIMPLIFIER_SAVE_DIR = "models/simplifier"
SUMMARIZER_SAVE_DIR = "models/summarizer"

# ===== 학습 하이퍼파라미터 =====
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 3e-4
