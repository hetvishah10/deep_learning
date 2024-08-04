import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    Y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    return x_train, x_test, y_train, y_test
