from src.symNN.sym_nn import SymNN

hyper_params = {'reg_change':0.3,
                'start_ln_block':1,
                'growth_steps':3,
                'l1_reg':0.1,
                'l2_reg':0.01,
                'num_epochs':200,
                'freeze':False}

est = SymNN(reg_change=0.3,
            start_ln_block=1,
            growth_steps=3,
            l1_reg=0.1,
            l2_reg=0.01,
            num_epochs=200,
            freeze=False)


def model(est):
    return est.recovered_eq

