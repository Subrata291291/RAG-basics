from src.chatbot import JewelleryChatbot


# ============================================================
# START CHATBOT
# ============================================================

def main():

    chatbot = JewelleryChatbot()

    print()
    print("=" * 60)
    print(
        "       JEWELLERY RAG CHATBOT"
    )
    print("=" * 60)

    print()
    print(
        "Products + Knowledge + "
        "Semantic Search + Conversation + Validation"
    )

    print()
    print(
        "Type 'exit' to quit."
    )

    while True:

        query = input(
            "\nAsk your jewellery question: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        if not query:

            continue

        try:

            answer = chatbot.ask(
                query
            )

            print()
            print(
                "===== FINAL ANSWER ====="
            )

            print(answer)

        except Exception as e:

            print()
            print(
                "ERROR:"
            )

            print(str(e))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()