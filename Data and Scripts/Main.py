import pandas as pd
from functools import partial
import multiprocessing
from CoreModel import single_task
import random

random.seed(2023)

def main_run(pool):
    dataset = pd.read_csv('Data/formatted_training_data.csv', header=0, index_col=0)
    dataset.drop("Date", axis=1, inplace=True)
    dataset.astype('float32')
    
    

    pool.map(partial(single_task, dataset=dataset, max_trials=10, batch_size=128, verbose=1,
                     epochs=10000, patience=200, doubleLayerLSTM=True), tasks)


if __name__ == '__main__':
    print('Starting the main function')

    n_in = 20
    m_out = 1
    tasks = [
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': False,
            'y_headers': ["WTI"],
            'x_headers': ["WTI", "henry", "Loughran-McDonald", "vader_average", "vader", "watson"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "absoluteCombination"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': False,
            'y_headers': ["WTI"],
            'x_headers': ["WTI", "vader_average", "vader"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "absolute_vader_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': False,
            'y_headers': ["WTI"],
            'x_headers': ["WTI", "henry", "Loughran-McDonald"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "absolute_henryLoughran_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': False,
            'y_headers': ["WTI"],
            'x_headers': ["WTI", "watson"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "absolute_watson_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': False,
            'y_headers': ["WTI"],
            'x_headers': ["WTI"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "absolute_NoSentiment_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': True,
            'y_headers': ["change_WTI_1"],
            'x_headers': ["WTI", "henry", "Loughran-McDonald", "vader_average", "vader", "watson"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "relative_combination_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': True,
            'y_headers': ["change_WTI_1"],
            'x_headers': ["WTI", "vader_average", "vader"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "relative_vader_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': True,
            'y_headers': ["change_WTI_1"],
            'x_headers': ["WTI", "henry", "Loughran-McDonald"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "relative_henryLoughran_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': True,
            'y_headers': ["change_WTI_1"],
            'x_headers': ["WTI", "watson"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "relative_watson_"+str(n_in)+"Periods"+str(m_out)
        },
        {
            'n_in': n_in,
            'm_out': m_out,
            'm_offset': 0,
            'relative': True,
            'y_headers': ["change_WTI_1"],
            'x_headers': ["WTI"],
            'training_split': 0.80,
            'rounds': 1000,
            'name': "relative_NoSentiment_"+str(n_in)+"Periods"+str(m_out)
        }
    ]

    project_name = "NeuralNetworks"

    # set the number of SFSR runs and the number of cores/processes to use for parallelization
    numProcesses = multiprocessing.cpu_count() # this is the maximum number of cores in your system

    numProcesses = min(numProcesses, len(tasks))
    numProcesses = 4
    print("number processes: ", numProcesses)
    pool = multiprocessing.Pool(processes=numProcesses)
    print('created multiprocessing pool')

    main_run(pool=pool)
