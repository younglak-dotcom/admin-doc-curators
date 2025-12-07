from transformers import pipeline

# 한국어 문장 순화: paraphrase/summarization 기반
simplifier = pipeline(
    "text2text-generation",
    model="paust/pko-t5-base"  # 한국어 T5 모델
)

def simplify_sentence(text):
    try:
        result = simplifier(text, max_length=80)[0]['generated_text']
        return result
    except Exception:
        return "문장 순화에 실패했습니다. 다시 시도해주세요!"

if __name__ == "__main__":
    print("행정문서 AI 순화 서비스")
    print("종료하려면 'q' 입력\n")

    while True:
        text = input("행정 문장을 입력하세요: ")

        if text.lower() == 'q':
            print("프로그램을 종료합니다.")
            break

        simplified = simplify_sentence(text)
        print("💡 쉬운 문장:", simplified)
        print()
