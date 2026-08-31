from src.chatbot import JewelleryChatbot


# ============================================================
# START CHATBOT
# ============================================================

def main():

    chatbot = JewelleryChatbot()

    print()
    print("=" * 60)
    print("       JEWELLERY RAG CHATBOT")
    print("=" * 60)

    print()
    print(
        "Products + Knowledge + Semantic Search + "
        "Conversation + Validation + Streaming"
    )

    print()
    print("Type 'exit' to quit.")

    while True:

        query = input(
            "\nAsk your jewellery question: "
        ).strip()

        if query.lower() == "exit":

            print("\nGoodbye!")

            break

        if not query:
            continue

        print()
        print("===== FINAL ANSWER =====")

        try:

            # =================================================
            # STREAMING RESPONSE
            # =================================================

            for chunk in chatbot.ask_stream(query):

                print(
                    chunk,
                    end="",
                    flush=True
                )

            print()

        except KeyboardInterrupt:

            print(
                "\n\nResponse interrupted."
            )

        except Exception as e:

            print()
            print("ERROR:")
            print(str(e))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()