from fruit_market.interface.nn_market import get_fruit_features, predict_fruit

def main():
    print("🍎 Fruit Preference Predictor 🍌")

    while True:
        features = get_fruit_features()
        print(f"\nPrediction: {'Like' if predict_fruit(features) else 'Dislike'}")

        if input("\nPredict another (y/n): ").lower() != 'y':
            print("Goodbye! 👋")
            break

if __name__ == "__main__":
    main()

