import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.model_selection import cross_val_score

class CustomerPredictionModel:

    def __init__(self, data_path):
        """
        Initialize the class with the data file path.

        Parameters:
        data_path (str): The path to the CSV data file.
        """
        self.df = pd.read_csv(data_path)

    def preprocess(self):
        """
        Preprocess the data. Converts the InvoiceDate column to datetime,
        removes rows with missing Customer ID and sorts the DataFrame by InvoiceDate.
        """
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df = self.df.dropna(subset=['Customer ID'])
        self.df = self.df.sort_values('InvoiceDate').reset_index(drop=True)

    def feature_engineering(self):
        """
        Perform feature engineering on the DataFrame. 
        Creates new features for the model.
        """
        self.df['PurchasedNextQuarter'] = ((self.df.sort_values('InvoiceDate').groupby('Customer ID', group_keys=True)['InvoiceDate']
                                            .shift(-1) - self.df['InvoiceDate']) <= pd.Timedelta(90, unit='D')).astype(int)
        self.df['HasReturned'] = self.df.groupby('Customer ID', group_keys=True)['Quantity'].apply(lambda x: (x < 0)).astype(int).values

        customer_data = self.df.groupby('Customer ID').agg({
            'Quantity': 'sum',
            'Price': 'mean',
            'InvoiceDate': ['min', 'max'], 
            'PurchasedNextQuarter': 'max', 
            'HasReturned': 'max' 
        })
        customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns.values]
        return customer_data

    def model_building(self, customer_data, params):
        """
        Build and train the model using the customer_data DataFrame and the specified hyperparameters.
    
        Parameters:
        customer_data (DataFrame): The customer data to use for model training.
        params (dict): Dictionary of hyperparameters to use in the model.
        """
        features = ['Quantity_sum', 'Price_mean', 'HasReturned_max']
        target = 'PurchasedNextQuarter_max'
        X = customer_data[features]
        y = customer_data[target]
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        scaler = StandardScaler()
        X_res = scaler.fit_transform(X_res)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
        self.model = RandomForestClassifier(**params, random_state=42)
        self.model.fit(X_train, y_train)
    
        return X_test, y_test

    def model_optimization(self, X_train, y_train):
        """
        Optimize the hyperparameters of the model using Optuna.
    
        Parameters:
        X_train (DataFrame): The train features.
        y_train (Series): The train target.
        """
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 10, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
            return cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
    
        return study.best_params


    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance by calculating accuracy, precision, recall, and ROC AUC.

        Parameters:
        X_test (DataFrame): The test features.
        y_test (Series): The test target.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nROC AUC: {roc_auc}')

    def plot_feature_importance(self):
        """
        Plot the feature importance of the RandomForest model.
        """
        plt.barh(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.yticks(range(len(self.model.feature_importances_)), ['Quantity_sum', 'Price_mean', 'HasReturned_max'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.show()

if __name__ == "__main__":
    model = CustomerPredictionModel("online_retail_II.csv")
    model.preprocess()
    customer_data = model.feature_engineering()

    # Define default parameters
    default_params = {'n_estimators': 469, 'max_depth': 22, 'min_samples_split': 3, 'min_samples_leaf': 2}
    X_test, y_test = model.model_building(customer_data, default_params)

    model.evaluate(X_test, y_test)
    model.plot_feature_importance()

    # Call optimization if you want to optimize the model
    # best_params = model.model_optimization(X_train, y_train)
    # print(f"Best parameters: {best_params}")
