from .de_learn_network import log_activation,\
    eql_model_v2, add_ln_block, set_model_l1_l2, L1L2_m
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocess_data(train_x, train_y=None, is_train=True, val_split=0.2):
    if is_train:
        train_count = int(train_x[0].shape[0] * (1 - val_split))
        val_x = train_x[:, train_count:]
        val_x = [x_in for x_in in val_x]
        train_x = train_x[:, :train_count]
        train_x = [x_in for x_in in train_x]
        val_y = train_y[train_count:]
        val_y = [val_y]
        train_y = train_y[:train_count]
        train_y = [train_y]
        return train_x, val_x, train_y, val_y
    else:
        train_x = [x_in for x_in in train_x]
        return train_x


def reg_stages_train(model, train_x, train_y, num_epochs, reg_change=0.3, l1_reg=1e-3, l2_reg=1e-3):
    stage1_epochs = int(reg_change*num_epochs)
    stage2_epochs = num_epochs - stage1_epochs
    train_x, val_x, train_y, val_y = preprocess_data(train_x, train_y)
    set_model_l1_l2(model, l1=0, l2=0)
    hist1 = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                      epochs=stage1_epochs, batch_size=32)
    print('intermediate weights', model.get_weights())
    set_model_l1_l2(model, l1=l1_reg, l2=l2_reg)
    hist2 = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                      epochs=stage2_epochs, batch_size=32)
    pred_y = model.predict(val_x)
    #print(val_y, pred_y)
    mse = mean_squared_error(val_y[0], pred_y)
    return model, [hist1, hist2], mse


def train_model_reg_stage(train_x, train_y, ln_block_count, num_epochs, decay_steps=1000, reg_change=0.3):
    print(train_x.shape)
    train_count = train_x.shape[1]
    x_dim = train_x.shape[0]
    reg_change1 = reg_change
    reg_change2 = 1
    # weight_thresh = 0.01
    stage1_epochs = int(reg_change1*num_epochs)
    stage2_epochs = int(reg_change2*num_epochs) - stage1_epochs
    # stage3_epochs = num_epochs - stage2_epochs
    model = eql_model_v2(input_size=x_dim, ln_block_count=ln_block_count,
                         decay_steps=train_count/10)
    model, history = reg_stages_train(model, train_x, train_y, num_epochs, reg_change)
    return model, history


def train_model_growth(train_x, train_y, start_ln_block, num_epochs, growth_steps=2, decay_steps=1000, freeze=False,
                       reg_change=0.3, l1_reg=1e-2, l2_reg=1e-2):
    # train_x = [np.array(x) for x in np.array(train_x.transpose())]
    #print(train_x)
    x_dim = len(train_x)
    cur_ln_blocks = start_ln_block
    model = eql_model_v2(x_dim, ln_block_count=start_ln_block, decay_steps=decay_steps)
    train_history = []
    print('model weights', model.get_weights())
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True)
    opt = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    # train_history.append(
    #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
    # )
    print(model.get_weights())
    model, history, mse = reg_stages_train(model, train_x, train_y, num_epochs, reg_change=reg_change,
                                      l1_reg=l1_reg, l2_reg=l2_reg)
    mse_cur = mse
    print('MSE', mse)
    actual_blocks = start_ln_block
    train_history += history
    for i in range(growth_steps):
        print(model.get_weights())
        tf.keras.utils.get_custom_objects().update({'log_activation':log_activation,
                                                    'L1L2_m':L1L2_m})
        model_new = tf.keras.models.clone_model(model)
        model_new.set_weights(model.get_weights())
        model_new = add_ln_block(x_dim, model_new, cur_ln_blocks, freeze_prev=freeze)
        print(model_new.get_weights())
        cur_ln_blocks += 1
        opt = optimizers.Adam(learning_rate=lr_schedule)
        model_new.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
        # train_history.append(
        #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
        # )
        model_new, history, mse = reg_stages_train(model_new, train_x, train_y, num_epochs,
                                                   0.3, l1_reg=l1_reg, l2_reg=l2_reg)
        print('MSE', mse)
        if mse > mse_cur*0.8:
            break
        model = tf.keras.models.clone_model(model_new)
        model.set_weights(model_new.get_weights())
        actual_blocks += 1
        mse_cur = mse
        train_history += history
    print(model.get_weights())
    return model, train_history, actual_blocks


