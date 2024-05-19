import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)


class Predictor:
    def __init__(self) -> None:

        self.models = {
            "Knn": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
        }

        self.all_errors = {
            "mae": {},
            "rmse": {},
            "mape": {},
        }

        for name, _ in self.models.items():
            self.all_errors["mae"][name] = {}
            self.all_errors["rmse"][name] = {}
            self.all_errors["mape"][name] = {}

    def predict_error(self, model, X_tr, X_ts, y_tr, y_ts):
        model = model.fit(X_tr, y_tr)
        test_predictions = model.predict(X_ts)
        mae = mean_absolute_error(y_ts, test_predictions)
        rmse = root_mean_squared_error(y_ts, test_predictions)
        mape = mean_absolute_percentage_error(y_ts, test_predictions)
        return mae, rmse, mape

    def calc_errors(self, feature_pair, X, y):
        n_folds = 10
        for name, model in self.models.items():
            model_errors = {
                "mae": [],
                "rmse": [],
                "mape": [],
            }
            kf = KFold(n_splits=n_folds)
            i = 0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                t_mae, t_rmse, t_mape = self.predict_error(
                    model, X_train, X_test, y_train, y_test
                )
                model_errors["mae"].append(t_mae)
                model_errors["rmse"].append(t_rmse)
                model_errors["mape"].append(t_mape)
                i += 1

            mae = np.mean(model_errors["mae"])
            rmse = np.mean(model_errors["rmse"])
            mape = np.mean(model_errors["mape"])
            self.all_errors["mae"][name][feature_pair] = mae
            self.all_errors["rmse"][name][feature_pair] = rmse
            self.all_errors["mape"][name][feature_pair] = mape

            print(name + " MAE:", mae)
            print(name + " RMSE:", rmse)
            print(name + " MAPE:", mape)
            print("----------------------------------")
