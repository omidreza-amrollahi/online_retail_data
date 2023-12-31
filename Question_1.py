import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import plotly.offline as py

class SalesForecaster:
    """
    This class provides methods for loading and preprocessing sales data, training a Prophet model,
    forecasting future sales, and saving the trained model.
    """

    def __init__(self, data_path):
        """
        Initialize the forecaster with the path to the sales data.
        """
        self.data_path = data_path
        self.model = None
        self.forecast = None

    def load_and_preprocess(self):
        """
        Load the data from a CSV file and preprocess it for Prophet.
        """
        df = pd.read_csv(self.data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.set_index('InvoiceDate', inplace=True)
        df['Sales'] = df['Quantity'] * df['Price']
        weekly_sales = df['Sales'].resample('W').sum()

        # Convert sales data to the format Prophet requires
        prophet_df = weekly_sales.reset_index()
        prophet_df.columns = ['ds', 'y']

        return prophet_df

    def train(self):
        """
        Train the Prophet model on the preprocessed data.
        """
        prophet_df = self.load_and_preprocess()
        self.model = Prophet()
        self.model.fit(prophet_df)

    def predict(self, periods):
        """
        Use the trained Prophet model to make predictions for the specified number of periods.
        """
        if self.model is None:
            raise Exception("You need to train the model before making predictions.")

        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)

        return self.forecast

    def save_model(self, path):
        """
        Save the trained Prophet model to the specified path.
        """
        if self.model is None:
            raise Exception("You need to train the model before saving it.")

        with open(path, 'w') as fout:
            fout.write(model_to_json(self.model))

    def plot_forecast(self):
        """
        Plot the forecasted values using the model's plot function.
        """
        if self.model is None or self.forecast is None:
            raise Exception("You need to train the model and make a prediction before plotting.")

        fig1 = plot_plotly(self.model, self.forecast)
        py.plot(fig1)
        fig2 = plot_components_plotly(self.model, self.forecast)
        py.plot(fig2)


# Example usage
forecaster = SalesForecaster('online_retail_II.csv')
forecaster.train()
forecast = forecaster.predict(4 * 7)  # Predict the next 4 weeks
forecaster.save_model('serialized_model.json')
forecaster.plot_forecast()