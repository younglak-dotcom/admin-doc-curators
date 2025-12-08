# data_stats.py

import pandas as pd
from collections import Counter
from config import PARALLEL_DATA_PATH

def load_parallel_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"source", "target", "domain"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV에 필요한 컬럼 {required_cols} 이(가) 없습니다.")
    return df

def compute_basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "num_pairs": len(df),
        "avg_source_len": df["source"].str.len().mean(),
        "avg_target_len": df["target"].str.len().mean(),
        "max_source_len": df["source"].str.len().max(),
        "max_target_len": df["target"].str.len().max(),
    }
    return stats

def domain_distribution(df: pd.DataFrame) -> Counter:
    return Counter(df["domain"])

if __name__ == "__main__":
    df = load_parallel_data(PARALLEL_DATA_PATH)

    stats = compute_basic_stats(df)
    print("📊 데이터 기본 통계")
    print(f"- 총 병렬 문장 쌍 수: {stats['num_pairs']:,}")
    print(f"- 원문 평균 길이: {stats['avg_source_len']:.1f}자 (최대 {stats['max_source_len']}자)")
    print(f"- 순화문 평균 길이: {stats['avg_target_len']:.1f}자 (최대 {stats['max_target_len']}자)\n")

    print("📂 도메인별 분포 (예: 정책/복지/법률/FAQ 등)")
    dist = domain_distribution(df)
    for domain, count in dist.most_common():
        ratio = count / stats["num_pairs"] * 100
        print(f"- {domain}: {count:,}개 ({ratio:.1f}%)")
