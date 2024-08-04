from sklearn.metrics import accuracy_score

def evaluate_model(model, x_train, y_train):
    loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

def make_predictions(model, x_test):
    y_preds = model.predict(x_test)
    y_preds = (y_preds > 0.5).astype(int).flatten()  # convert probabilities to binary outcomes
    return y_preds