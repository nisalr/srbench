from symnn.sym_nn import SymNN
from sympy import preorder_traversal

hyper_params = {'reg_change':0.5,
                'start_ln_blocks':1,
                'growth_steps':3,
                'l1_reg':1e-4,
                'l2_reg':1e-4,
                'num_epochs':500,
                'round_digits':3,
                'train_iter':4}

est = SymNN(reg_change=0.5,
            start_ln_blocks=1,
            growth_steps=3,
            l1_reg=1e-4,
            l2_reg=1e-4,
            num_epochs=500,
            round_digits=3,
            train_iter=4)

def eq_complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c

def model(est):
    return est.recovered_eq


def complexity(est):
    eq = est.recovered_eq
    return eq_complexity(eq)


