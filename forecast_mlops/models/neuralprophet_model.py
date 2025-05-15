"""
Create a Neural Prophet model for forecasting day-ahead load and generation
"""
from typing import Literal, Optional, Tuple, Union

from neuralprophet import NeuralProphet
import pandas as pd

import data.constants as c


def train_neuralprophet_model(
    profile: pd.DataFrame,
    input_name: str,
    timestamp: str,
    daily_seasonality: Union[int, Literal["auto", True, False]],
    weekly_seasonality: Union[bool, Literal["auto", True, False]],
    yearly_seasonality: Union[bool, Literal["auto", True, False]],
    n_lags: Optional[int] = None,
    growth: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_hidden_layers: Optional[int] = None,
    d_hidden: Optional[int] = None,
    country_holidays: Optional[str] = None,
    ar_reg: Optional[float] = None,
    epochs: Optional[int] = None,
) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.Series], Optional[NeuralProphet]
        ]:
    """
    Train a NeuralProphet model on historical data and product multi hours
    forecasts.
    Args:
        profile: an array of timeseries data for a bus.
        input_name: a variable name of the model input.
        timestamp: Timestamp
        daily_seasonality: seasonal components to be modelled.
        weekly_seasonality: seasonal components to be modelled.
        yearly_seasonality: seasonal components to be modelled.
        n_lags: A number of lags determines how far into the past
            the auto-regressive dependencies should be considered.
            If n_lags > 0, the AR-Net is enabled.
        growth: trend growth type
        learning_rate: learning rate
        num_hidden_layers: a number that defines the number of hidden layers
            of the FFNNs used in the overall model.
        d_hidden: the number of units in the hidden layers.
        country_holidays: a list of country specific holidays.
        ar_reg: how much sparsity to induce in the AR-coefficients.
        epochs: Number of epochs (complete iterations over dataset) to
            train model.
    """

    prophet_data = (
        profile.reset_index()[[timestamp, input_name]]
        .rename(columns={timestamp: "ds", input_name: "y"})
        .copy()
    )

    # Define training and testing datasets
    df_train = prophet_data.drop(prophet_data.index[-c.HOURS:])
    df_test = prophet_data[-c.HOURS:]

    neural_forecaster = NeuralProphet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        n_lags=n_lags,
        growth=growth,
        future_regressors_num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
        future_regressors_d_hidden=d_hidden,
        ar_reg=ar_reg,
        epochs=epochs,
    )
    if country_holidays is not None:
        neural_forecaster.add_country_holidays(country_name=country_holidays)

    # fit the model
    neuralprophet_model_metrics = neural_forecaster.fit(
        df_train,
        freq="H",
        validation_df=df_test
    ).tail(1)

    # make a prediction
    forecast = neural_forecaster.predict(df_test)
    return (
        forecast[['ds', 'y']].set_index('ds'),
        neuralprophet_model_metrics,
        neural_forecaster,
    )
