from transformers import pipeline

# 한국어 문장 순화 모델
simplifier = pipeline(
    "text2text-generation",
    model="paust/pko-t5-base",  # 쉬운 문장으로 재작성
    max_length=128
)

# 한국어 요약 모델 (정확도 우선)
summarizer = pipeline(
    "summarization",
    model="psyche/KoT5-summarization",  # 정확도 우선
    max_length=60
)

def simplify_sentence(text: str) -> str:
    try:
        result = simplifier(text)[0]['generated_text']
        return result.strip()
    except Exception:
        return "문장 순화에 실패했습니다."

def summarize_sentence(text: str) -> str:
    try:
        result = summarizer(text)[0]['summary_text']
        return result.strip()
    except Exception:
        return "요약에 실패했습니다."

if __name__ == "__main__":
    print("📘 행정문서 AI 순화 서비스")
    print("종료하려면 'q' 입력\n")

    while True:
        text = input("행정 문장을 입력하세요: ")

        if text.lower() == 'q':
            print("프로그램을 종료합니다.")
            break

        simplified = simplify_sentence(text)
        summary = summarize_sentence(text)

        print("\n💡 쉬운 문장:")
        print(simplified)

        print("\n📌 핵심 요약:")
        print(summary)
        print("\n" + "="*50 + "\n")
