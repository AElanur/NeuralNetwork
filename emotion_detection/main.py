from .data.data_loader import DataLoader

def main():
    data_loader = DataLoader()

    while True:
        user_input = input("You: ")

        greeting_response = data_loader.handle_greeting(user_input)
        if greeting_response:
            print(f"Bot: {greeting_response}")
            continue

        if response := data_loader.handle_expressions(user_input):
            print(f"Bot: {response}")
            continue

if __name__ == "__main__":
    main()
