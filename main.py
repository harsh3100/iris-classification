from data.load_data import load_iris_data
from utils.helpers import split_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualize import plot_pairplot

def main():
    df, iris = load_iris_data()
    plot_pairplot(df)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)

    print(f"\nModel Accuracy: {acc:.2f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
