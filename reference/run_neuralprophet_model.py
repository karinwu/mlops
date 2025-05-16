"""
Train a Neural Prophet model for forecasting day-ahead load and generation
"""
import os

from argparse import ArgumentParser
import mlflow
from omegaconf import OmegaConf
import optuna
import pandas as pd

import data.constants as c
from forecast_mlops.mlflow_tracker import (
    MLFlowTracker,
)
from forecast_mlops.models.neuralprophet_model import (
    train_neuralprophet_model,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOURS = 24

METRIC_LIST = ["MAE", "RMSE", "MAE_val", "RMSE_val"]

PARAM_BOUNDS = {
    "n_lags": [1, 12],
    "learning_rate": [0.001, 0.01],
    "num_hidden_layers": [1, 2],
    "d_hidden": [1, 4],
    "daily_seasonality": [True, False],
    "weekly_seasonality": [True, False],
    "yearly_seasonality": [True, False],
    # "growth": ["off", "linear"],
    "growth": ["off"],
    "country_holidays": [None, "US"],
    "ar_reg": [0.0001, 1],
}
EXPERIMENT_NAME = "neuralprophet-optuna-mlflow-test"
MODEL_CLASS = mlflow.pytorch.log_model


def objective(
    trial: optuna.trial.Trial,
    experiment_name: str,
    max_epochs: int,
) -> float:
    tracker = MLFlowTracker(
        experiment_name=experiment_name,
        tracking_uri=conf["tracking_uri"],
        to_track=conf["to_track"],
    )

    buses = pd.read_csv(
        os.path.join(BASE_DIR, 'reference', 'demand_hourly.csv')
    )
    buses[c.TIMESTAMP] = pd.to_datetime(buses[c.TIMESTAMP])

    buses = buses[buses.BusType.isin(c.LOAD_GEN_LIST)]
    bus_id = buses.Bus.unique()
    output_data = pd.DataFrame()
    output_metrics = pd.DataFrame()

    with mlflow.start_run():
        model_parameters = {}
        for param in ["n_lags", "num_hidden_layers", "d_hidden"]:
            model_parameters[param] = trial.suggest_int(
                param, PARAM_BOUNDS[param][0], PARAM_BOUNDS[param][1]
            )
        for param in ["learning_rate", "ar_reg"]:
            model_parameters[param] = trial.suggest_float(
                param, PARAM_BOUNDS[param][0], PARAM_BOUNDS[param][1], log=True
            )
        for param in [
            "daily_seasonality",
            "weekly_seasonality",
            "yearly_seasonality",
            "growth",
            "country_holidays",
        ]:
            model_parameters[param] = trial.suggest_categorical(
                param, PARAM_BOUNDS[param]
            )

        # iterate over Buses
        for bus in bus_id.tolist():
            bus_profile = buses[buses.Bus == bus][[c.TIMESTAMP, c.INPUT_NAME]]
            (
                model_results,
                neuralprophet_model_metrics,
                neural_forecaster,
            ) = train_neuralprophet_model(
                bus_profile,
                input_name=c.INPUT_NAME,
                timestamp=c.TIMESTAMP,
                epochs=max_epochs,
                **model_parameters,
            )
            model_results["Bus"] = bus
            neuralprophet_model_metrics["Bus"] = bus

            output_metrics = pd.concat(
                [output_metrics, neuralprophet_model_metrics],
                ignore_index=True
            )
            output_data = pd.concat(
                [output_data, model_results],
                ignore_index=True
            )

            metrics = output_metrics[METRIC_LIST].mean()
            metrics_dict = {keys: metrics[keys] for keys in METRIC_LIST}

            tracker.log_and_end(
                model=neural_forecaster.model,
                model_logger=MODEL_CLASS,
                metrics=metrics_dict,
                params=trial.params,
            )
        return output_metrics["RMSE"].mean()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml_path",
        type=str,
        metavar="",
        help="path to yaml configuration file",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type=int,
        default=2,
        help="Number of trials.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=6000,
        help="Stop study after the given number of seconds.",
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=2,
        help="Max number of epochs to run model for.",
    )
    args = parser.parse_args()
    config_path = args.config_yaml_path

    conf = OmegaConf.load(config_path)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=EXPERIMENT_NAME,
        direction="minimize"
    )
    study.optimize(
        lambda trial: objective(
            trial,
            experiment_name=EXPERIMENT_NAME,
            max_epochs=args.max_epochs,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
