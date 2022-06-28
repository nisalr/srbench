from .data_gen  import linear_2D_eq, data_gen_eq, div_eq, div_add, lorenz_eq, data_gen_de, mult_eq, pow_eq, \
    cube_add_eq, linear_3D_eq, get_symbolic_fn
from .de_learn_network import eql_model, log_activation, eql_model_signed, eql_model_v2, set_model_l1_l2, L1L2_m, \
    WeightMaskCallback
import numpy as np
import mlflow
import time

from matplotlib import pyplot as plt
from .train_model import train_model_growth, train_model_reg_stage

from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({ L1L2_m.__name__: L1L2_m, log_activation.__name__: log_activation })
from utils import get_sympy_expr_v2

if __name__ == '__main__':
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-12

    # eq_names = ['linear_2D_eq', 'pow_eq', 'div_eq',
    #             'div_add', 'mult_eq', 'cube_add_eq']

    eq_names = ['comb_3D']

    req_blocks = {'linear_2D_eq': 2,
                  'pow_eq': 2,
                  'div_eq': 1,
                  'div_add': 3,
                  'mult_eq': 1,
                  'cube_add_eq': 2,
                  'pow_3D': 3,
                  'comb_3D': 2}

    repeat_count = 3
    # repeat experiments multiple times with different initializations
    eq_names = [item for item in eq_names for i in range(repeat_count)]
    for eq_name in eq_names:
        x_dim = 3
        # eq_name = 'cube_add_eq'
        eq_func, eq_expr = get_symbolic_fn(eq_name)
        ln_block_count = 3
        train_count = 100000
        growth_steps = 0
        input_range = (1, 10)
        num_epochs = 200
        l1_reg = 1e-1
        l2_reg = 1e-2
        reg_change = 0.3
        start_ln_block = req_blocks[eq_name]

        mlflow.set_tracking_uri('../mlruns/')
        mlflow.set_experiment('de_learn_nn')
        mlflow.start_run()
        mlflow.log_param('input dimensions', x_dim)
        mlflow.log_param('input range', input_range)
        mlflow.log_param('ln_block_count', ln_block_count)
        mlflow.log_param('start ln blocks', start_ln_block)
        mlflow.log_param('growth_steps', growth_steps)
        mlflow.log_param('equation name', eq_name)
        mlflow.log_param('equation expression', eq_expr)
        mlflow.log_param('train count', train_count)
        mlflow.log_param('epoch count', num_epochs)
        mlflow.log_param('regularization change', reg_change)
        mlflow.log_param('L1 reg', l1_reg)
        mlflow.log_param('L2 reg', l2_reg)

        train_x, train_y, test_x, test_y = data_gen_eq(eq_func, input_range, x_dim, train_count, validation_split=0.2)
        # , train_x, train_y = data_gen_de(lorenz_eq, 0.1, 10000,
        #                                             [1, 2, 3], integrator_keywords)
        print('train x shape', len(train_x))
        print(train_x)

        start_time = time.time()
        model, train_history, blk_count = train_model_growth(train_x, train_y, start_ln_block,
                                                             num_epochs, growth_steps, freeze=False,
                                                             l1_reg=l1_reg, l2_reg=l2_reg)
        run_time = time.time() - start_time

        val_mse = []
        for history in train_history:
            plt.figure()
            val_mse += history.history['val_mean_squared_error']
        fig = plt.figure()
        plt.plot(val_mse)
        plt.yscale('log')
        plt.show()
        mlflow.log_figure(fig, 'validation_error.png')

        model.save('../models/model_' + eq_func.__name__ + '.h5')

        pred_y = model.predict(test_x)
        print(pred_y, test_y[0])
        mse_test = mean_squared_error(test_y[0], pred_y)
        print('mean sq err', mse_test)
        pred_y_train = model.predict(train_x)
        mse_train = mean_squared_error(train_y[0], pred_y_train)
        print('mean sq err train', mse_train)
        mlflow.log_metrics({'MSE test':mse_test,
                            'MSE train':mse_train,
                            'train time':run_time,
                            'block_count':blk_count})
        for layer in model.layers:
            print(layer.name, layer.get_weights())
        print()
        # print(model.get_weights())
        # print(model.get_layer('output_dense').get_weights())
        recovered_eq = get_sympy_expr_v2(model, x_dim, blk_count, round_digits=2)
        mlflow.log_param('recovered_eq.txt', recovered_eq)
        mlflow.end_run()

