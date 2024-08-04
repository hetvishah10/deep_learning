import tensorflow as tf
from src.data.make_dataset import load_data, preprocess_data
from sklearn.metrics import accuracy_score 
from src.models.train_model import (
    build_model_1, build_model_2, build_model_3, build_model_4, build_model_5, compile_and_train_model
)
from src.models.predict_model import evaluate_model, make_predictions
from src.visualization.visualize import plot_learning_rate_vs_loss, plot_training_curves

def main():
    # Load and preprocess the data
    file_path = 'data/raw/employee_attrition.csv'
    df = load_data(file_path)
    x_train, x_test, y_train, y_test = preprocess_data(df)
    
    # Build and train the model
    model_1 = build_model_1()
    model_1, history_1 = compile_and_train_model(model_1, x_train, y_train, epochs=5)
    
    # Evaluate the model
    print("Model 1 Evaluation:")
    evaluate_model(model_1, x_train, y_train)
    
    # Predict and evaluate on the test set
    y_preds = make_predictions(model_1, x_test)
    print("Accuracy on test set:", accuracy_score(y_test, y_preds))
    
    # Plot learning rate vs. loss
    plot_learning_rate_vs_loss(history_1)

    
    # Plot the training curves
    plot_training_curves(history_1)

if __name__ == "__main__":
    main()
