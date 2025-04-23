from interface.ml_market import ppn, fruit_names, get_fruit_features

def main():
    print("🍎 Fruit Preference Predictor 🍌")
    print("Trained weights:", ppn.w_)

    while True:
        features = get_fruit_features()
        prediction = ppn.predict(features.reshape(1, -1))[0]

        print(f"\nPrediction: {fruit_names[prediction]}")

        if input("\nPredict another (y/n): ").lower() != 'y':
            print("Goodbye! 👋")
            break

if __name__ == "__main__":
    main()

