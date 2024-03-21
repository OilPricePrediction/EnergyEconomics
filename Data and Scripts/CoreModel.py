import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
from matplotlib import pyplot
import numpy as np
import time
import joblib
import kerastuner as kt
from sklearn.model_selection import TimeSeriesSplit
import os.path

np.random.seed(2023)

# the following code is necessary to parallelize on a GPU instance
# if the code is CPU-parallelized, it will be ignored
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def calculate_profit(prepared_dataset, y_hat, relative):
    barrels = 0.0
    value = 0.0
    profit = 0.0
    max_money = 0.0
    for i in range(len(y_hat)):
        price = prepared_dataset["WTI_t-1"][i]
        prediction = y_hat[i][0]
        if (((not relative) and prediction > price) or (relative and prediction > 0)):
            barrels += 1
            value += price
            if (value > max_money):
                max_money = value
        else:
            profit += price * barrels - value
            barrels = 0.0
            value = 0.0

    return ({'profit': profit, 'max_money': max_money})


def series_to_supervised(data, x, y, n_in=1, m_out=1, m_offset=0, dropnan=True ,):
    df = pd.DataFrame(data, copy=True)
    cols = {}

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        shifted = df.shift(i)
        for header in x:
            cols[f'{header}_t-{i}'] = shifted[header].to_numpy()
    # forecast sequence (t, t+1, ... t+m)
    for i in range( 0 -m_offset, m_out -m_offset):
        shifted = df.shift(-i)
        for header in y:
            if i >= 0:
                cols[f'{header}_t{f"+{i}" if i > 0 else ""}'] = shifted[header].to_numpy()
            else:
                cols[f'{header}_t{i}'] = shifted[header].to_numpy()

    df_out = pd.DataFrame(cols)
    if dropnan:
        df_out.dropna(inplace=True)
        df_out.reset_index(inplace=True)
        df_out.drop(labels="index", axis=1, inplace=True)

    df_out.reset_index()
    return df_out


def single_task(task, dataset, n_splits=5, max_trials=30, batch_size=32, verbose=1, epochs=10000, patience=2000,
                doubleLayerLSTM=True):
    n_in = task['n_in']
    m_out = task['m_out']
    m_offset = task['m_offset']
    relative = task['relative']
    y_headers = task['y_headers']
    x_headers = task['x_headers']
    training_split = task['training_split']
    rounds = task['rounds']

    print(task)

    if relative:
        dataset.change_WTI_1 = (dataset.WTI.shift(-1) - dataset.WTI).shift(1)

    prepared_dataset = series_to_supervised(dataset, x_headers, y_headers, n_in=n_in, m_out=m_out, m_offset=m_offset)

    split = math.floor(len(dataset) * training_split)

    test_size = len(dataset) - split
    test_start_index = len(prepared_dataset) - test_size

    val_split = 0.6
    split_val = math.floor(len(dataset) * val_split)

    train = prepared_dataset.iloc[:split_val, ]
    val = prepared_dataset.iloc[split_val:test_start_index, :]
    test = prepared_dataset.iloc[test_start_index:, :]

    # update
    scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_x = scaler_x.fit_transform(train.iloc[:, :-len(y_headers) * m_out])
    val_x = scaler_x.transform(val.iloc[:, :-len(y_headers) * m_out])
    test_x = scaler_x.transform(test.iloc[:, :-len(y_headers) * m_out])

    scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_y = scaler_y.fit_transform(train.iloc[:, -len(y_headers) * m_out:])
    val_y = scaler_y.transform(val.iloc[:, -len(y_headers) * m_out:])
    test_y = scaler_y.transform(test.iloc[:, -len(y_headers) * m_out:])

    train_x = train_x.reshape((train_x.shape[0], n_in, len(x_headers)))
    val_x = val_x.reshape((val_x.shape[0], n_in, len(x_headers)))
    test_x = test_x.reshape((test_x.shape[0], n_in, len(x_headers)))

    train_x_full = np.concatenate((train_x, val_x))
    train_y_full = np.concatenate((train_y, val_y))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def build_model(hp):
        model = models.Sequential()

        if doubleLayerLSTM:
            units = hp.Int('units', min_value=2, max_value=15, step=1)
            model.add(layers.LSTM(units,
                                  input_shape=(n_in, len(x_headers)),
                                  kernel_initializer=hp.Choice('activation', values=['normal', 'glorot_uniform']),
                                  return_sequences=True))
            model.add(layers.Dropout(rate=hp.Choice('rate', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])))
            #     model.add(layers.LSTM(layer_out, dropout=dropout))
            model.add(layers.LSTM(units,
                                  input_shape=(n_in, len(x_headers)),
                                  kernel_initializer=hp.Choice('activation', values=['normal', 'glorot_uniform'])))
            model.add(layers.Dropout(rate=hp.Choice('rate', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])))

            model.add(layers.Dense(train_y.shape[1], kernel_initializer=hp.Choice('activation', values=['glorot_uniform'])))
        else:
            units = hp.Int('units', min_value=2, max_value=15, step=1)
            model.add(layers.LSTM(units,
                                  input_shape=(n_in, len(x_headers)),
                                  kernel_initializer=hp.Choice('activation', values=['normal', 'glorot_uniform'])))
            model.add(layers.Dropout(rate=hp.Choice('rate', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])))
            #     model.add(layers.LSTM(layer_out, dropout=dropout))
            model.add(layers.Dense(train_y.shape[1], kernel_initializer=hp.Choice('activation', values=['glorot_uniform'])))

        model.compile(loss='mse', optimizer=hp.Choice('optimizer', values=['adam', 'RMSprop']),
                      metrics=['mean_squared_error'])
        return model


    class CVTuner(kt.engine.tuner.Tuner):
        def run_trial(self, trial, x, y, batch_size=32, epochs=10000, callbacks=None):
            #     cv = model_selection.KFold(5)
            cv = tscv
            val_losses = []
            for train_indices, test_indices in cv.split(x):
                x_train, x_test = x[train_indices], x[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                model = self.hypermodel.build(trial.hyperparameters)
                model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                          callbacks=callbacks, verbose=verbose)
                val_losses.append(model.evaluate(x_test, y_test))
            self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
            self.save_model(trial.trial_id, model)


    tuner = CVTuner(
        hypermodel=build_model,
        #   oracle=kt.oracles.BayesianOptimization(
        #     objective='val_loss',
        #     max_trials=3),
        oracle=kt.oracles.RandomSearch(
            objective='val_loss',
            max_trials=max_trials),
        overwrite=True,
        project_name=task['name']
    )

    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',
                                         restore_best_weights=True)

    tuner.search(train_x_full, train_y_full, batch_size=batch_size, epochs=epochs, callbacks=[stop_early])

    best_model = tuner.get_best_models()[0]

    train_pred = best_model.predict(train_x_full)
    # for the test set
    test_pred = best_model.predict(test_x)

    # re-training on the entire training data using the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]

    hypermodel = tuner.hypermodel.build(best_hp)
    #
    # Retrain the model
    hypermodel.fit(train_x_full, train_y_full, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                   callbacks=[stop_early])
    #
    test_pred_hm = hypermodel.predict(test_x)

    # calculate the RMSE
    # note: this depends on if it's an absolute or relative prediction

    # for the training set
    if relative:
        y_hat_train = scaler_y.inverse_transform(train_pred)

        temp_data = prepared_dataset.iloc[:test_start_index, ]
        y_hat_train_final = temp_data['WTI_t-1'] + y_hat_train.flatten()
        #
        model_rmse_train = math.sqrt(metrics.mean_squared_error(dataset.WTI[:test_start_index:], y_hat_train_final))

        y_hat_train_final = y_hat_train_final.to_list()

        # we  add these predictions to the current WTI values to get the final predictions
        y_hat = scaler_y.inverse_transform(test_pred)
        y_hat_hm = scaler_y.inverse_transform(test_pred_hm)

        y_hat_final = test['WTI_t-1'] + y_hat.flatten()
        y_hat_hm_final = test['WTI_t-1'] + y_hat_hm.flatten()
        y_hat_final = y_hat_final.to_list()
        y_hat_hm_final = y_hat_hm_final.to_list()

        # model_rmse = math.sqrt(metrics.mean_squared_error(dataset.WTI[split+n_in:], y_hat_final))
        # model_rmse_hm = math.sqrt(metrics.mean_squared_error(dataset.WTI[split + n_in:], y_hat_hm_final))
        model_rmse = math.sqrt(metrics.mean_squared_error(dataset.WTI[test_start_index+n_in:], y_hat_final))
        model_rmse_hm = math.sqrt(metrics.mean_squared_error(dataset.WTI[test_start_index+n_in:], y_hat_hm_final))

    else:
        y_hat_train_final = scaler_y.inverse_transform(train_pred)

        y_train = scaler_y.inverse_transform(train_y_full)

        model_rmse_train = math.sqrt(metrics.mean_squared_error(y_train, y_hat_train_final))

        y_hat_final = scaler_y.inverse_transform(test_pred)
        y_hat_hm_final = scaler_y.inverse_transform(test_pred_hm)
        y_test = scaler_y.inverse_transform(test_y)

        model_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_hat_final))
        model_rmse_hm = math.sqrt(metrics.mean_squared_error(y_test, y_hat_hm_final))

    results_col = ['optimizer', 'init', 'layer_out', 'patience', 'batch_size', 'dropout', 'rmse',
                   'rmse_hypermodel_retrained', 'rmse_train', 'y_hat', 'y_hat_hypermodel_retrained', 'y_hat_train']
    results_col_short = ['optimizer', 'init', 'layer_out', 'patience', 'batch_size', 'dropout', 'rmse',
                   'rmse_hypermodel_retrained', 'rmse_train', 'y_hat', 'y_hat_hypermodel_retrained']
    # check if a previous file for the same scenario exists. If yes, add the results to the file, if not, create it
    save_name = 'OverallModelResults_Task_' + task['name'] + 'DoubleLayer' + str(doubleLayerLSTM) + '.csv'

    file_path = os.path.abspath(save_name)
    if os.path.exists(file_path):
        model_results = pd.read_csv(save_name)
        # the first column will be the index, which can be dropped.
        # model_results = model_results.drop(model_results.columns[0], axis=1)
        if not 'y_hat_train' in model_results.columns:
            model_results = pd.read_csv(save_name, usecols=results_col_short)
            model_results['y_hat_train'] = "0"
        else:
            model_results = pd.read_csv(save_name, usecols=results_col)

    else:
        model_results = pd.DataFrame(columns=results_col)

    best_hp = tuner.get_best_hyperparameters()[0]

    model_results.loc[len(model_results)] = [best_hp['optimizer'], best_hp['activation'], best_hp['units'], patience,
                                             batch_size, best_hp['rate'], model_rmse, model_rmse_hm,
                                             model_rmse_train, y_hat_final, y_hat_hm_final, y_hat_train_final]
    model_results.to_csv(save_name)

    