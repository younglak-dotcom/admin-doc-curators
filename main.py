# main.py
"""
행정문서 AI 큐레이터 통합 데모
- 행정 문장 순화
- 핵심 요약
- 관련 행정 문서 큐레이션
"""

from transformers import pipeline
from config import SIMPLIFIER_SAVE_DIR, SUMMARIZER_SAVE_DIR
from curator import AdminDocCurator

def load_pipelines():
    simplifier = pipeline(
        "text2text-generation",
        model=SIMPLIFIER_SAVE_DIR,  # 파인튜닝된 모델 경로
        max_length=128,
    )
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_SAVE_DIR,  # 없으면 base 모델 쓰도록 바꿔도 됨
        max_length=60,
    )
    return simplifier, summarizer

def main():
    print("📘 행정문서 AI 큐레이터 (FULL PIPELINE)")
    print("종료하려면 'q' 입력\n")

    simplifier, summarizer = load_pipelines()
    curator = AdminDocCurator()

    while True:
        text = input("행정 문장 또는 질문을 입력하세요: ")

        if text.lower() == "q":
            print("프로그램을 종료합니다.")
            break

        # 순화
        simple = simplifier(text)[0]["generated_text"].strip()
        # 요약
        summary = summarizer(text)[0]["summary_text"].strip()
        # 큐레이션
        recs = curator.recommend(text, top_k=3)

        print("\n💡 쉬운 문장:")
        print(simple)

        print("\n📌 핵심 요약:")
        print(summary)

        print("\n📎 관련 행정 문서 추천:")
        for doc_id, score in recs:
            print(f"- {doc_id} (유사도 {score:.3f})")

        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
