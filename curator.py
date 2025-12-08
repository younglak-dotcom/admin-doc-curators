# curator.py
"""
행정문서 큐레이터
- 사용자의 질문/문장에 대해 유사한 행정문서 추천
"""

from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import ADMIN_DOCS_PATH

class AdminDocCurator:
    def __init__(self, csv_path: str = ADMIN_DOCS_PATH):
        self.df = pd.read_csv(csv_path)  # doc_id, text, domain
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.doc_matrix = self.vectorizer.fit_transform(self.df["text"])

    def recommend(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            doc_id = self.df.iloc[i]["doc_id"]
            score = float(sims[i])
            results.append((doc_id, score))
        return results

if __name__ == "__main__":
    curator = AdminDocCurator()
    q = input("질문을 입력하세요: ")
    recs = curator.recommend(q)
    print("\n📎 관련 문서 추천 결과")
    for doc_id, score in recs:
        print(f"- {doc_id} (유사도 {score:.3f})")
