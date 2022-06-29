from .train_model import train_model_growth, preprocess_data
from .utils import get_sympy_expr_v2
from .de_learn_network import log_activation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
import numpy as np

class SymNN(BaseEstimator, RegressorMixin):

    def __init__(self, reg_change=0.3, start_ln_block=1, growth_steps=2,
                 l1_reg=0.1, l2_reg=0.01, num_epochs=200, freeze=False):
        self.reg_change = reg_change
        self.start_ln_block = start_ln_block
        self.growth_steps = growth_steps
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.freeze = freeze
        self.num_epochs = num_epochs

    def fit(self, X, y):
        print(len(y))
        print(X, y)
        X, y = check_X_y(X, y)
        X = np.array(X).transpose()
        x_dim = X.shape[0]
        model, train_history, blk_count = train_model_growth(
            X, y, self.start_ln_block, self.num_epochs, self.growth_steps, freeze=self.freeze,
            l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        self.model = model
        self.train_history = train_history
        self.blk_count = blk_count

        recovered_eq = get_sympy_expr_v2(model, x_dim, blk_count, round_digits=2)
        self.recover_eq = recovered_eq
        return self

    def predict(self, X):
        X = np.array(X).transpose()
        train_x = preprocess_data(X, is_train=False)
        pred_y = self.model.predict(train_x)
        print(pred_y)
        return pred_y

if __name__ == '__main__':
    check_estimator(SymNN())




