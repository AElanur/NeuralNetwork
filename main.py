# from interface.ml_market import ppn, fruit_names, get_fruit_features
from interface.dl_market import get_fruit_features, fruit_names, predict_fruit

def main():
    print("🍎 Fruit Preference Predictor 🍌")
    # print("Trained weights:", ppn.w_)

    while True:
        features = get_fruit_features()
        pred_class = 1 if predict_fruit(features) == "Like" else 0

        print(f"\nPrediction: {fruit_names[pred_class]}")

        if input("\nPredict another (y/n): ").lower() != 'y':
            print("Goodbye! 👋")
            break

        # prediction = ppn.predict(features.reshape(1, -1))[0]
        #
        # print(f"\nPrediction: {fruit_names[prediction]}")
        #
        # if input("\nPredict another (y/n): ").lower() != 'y':
        #     print("Goodbye! 👋")
        #     break

if __name__ == "__main__":
    main()

