# Forecasting with MLOps
This repo is a tool for structuring, tracking data science and machine learning projects.

The mlflow tracker contains a wrapper class for using `MLflow` tracking functionality. It initializes runs and performs different logging functionalities.

## Short Term Forecasting
This component includes methodology used for forecasting the short term load (day-ahead) and generation in an hourly resolution.

`NeuralProphet` is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by [Meta Neural Prophet](https://ai.meta.com/blog/neuralprophet-the-neural-evolution-of-facebooks-prophet/) and AR-Net.
