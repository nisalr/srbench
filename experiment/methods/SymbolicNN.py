from .src.symNN.sym_nn import SymNN
from sympy import preorder_traversal

hyper_params = {'reg_change':0.3,
                'start_ln_block':1,
                'growth_steps':3,
                'l1_reg':0,
                'l2_reg':0,
                'num_epochs':500,
                'freeze':False,
                'round_digits':3}

est = SymNN(reg_change=0.3,
            start_ln_block=1,
            growth_steps=3,
            l1_reg=0,
            l2_reg=0,
            num_epochs=500,
            freeze=False,
            round_digits=3)

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


