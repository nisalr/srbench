import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols


def get_symbolic_fn(eq_name):
    func_dict = {'linear_2D_eq': (linear_2D_eq, 2*symbols('x1') + 3*symbols('x2')),
                 'pow_eq': (pow_eq, symbols('x1')**2 + symbols('x2')**2),
                 'linear_3D_eq': (linear_3D_eq, 2*symbols('x1') + 3*symbols('x2') + 4*symbols('x3')),
                 'div_eq': (div_eq, symbols('x1')/symbols('x2')),
                 'div_add': (div_add, symbols('x1')/symbols('x2') + 2*symbols('x1') - 5*symbols('x2')),
                 'mult_eq': (mult_eq, symbols('x1')*symbols('x2')),
                 'cube_add_eq': (cube_add_eq, symbols('x1')**3 + symbols('x2')),
                 'pow_3D': (pow_3D, symbols('x1')**3 + symbols('x2')**2 + symbols('x3')),
                 'comb_3D': (comb_3D, symbols('x3') * symbols('x2') + symbols('x2') * symbols('x1'))}
    return func_dict[eq_name]


def linear_2D(t, x):
    x1, x2 = x[0], x[1]
    x1_dot = -0.1 * x1 + 2 * x2
    x2_dot = -2 * x1 - 0.1 * x2
    return [
        x1_dot, x2_dot
    ]


def cubic_damped_SHO(t, x):
    return [
        -0.1 * x[0] ** 3 + 2 * x[1] ** 3,
        -2 * x[0] ** 3 - 0.1 * x[1] ** 3,
    ]


def lorenz_eq(t, x):
    x, y, z = x[0], x[1], x[2]
    x_dot = x - y
    y_dot = 2*x - x*z - y
    z_dot = x*y - 3*z
    return [x_dot, y_dot, z_dot]


def pow_3D(x):
    x1, x2, x3 = x[0], x[1], x[2]
    y1 = x1**3 + x2**2 + x3
    return [y1]


def comb_3D(x):
    x1, x2, x3 = x[0], x[1], x[2]
    y1 = x3 * x2 + x2 * x1
    return [y1]


def linear_2D_eq(x):
    x1, x2 = x[0], x[1]
    y1 = 2*x1 + 3*x2
    return [y1]

def linear_3D_eq(x):
    x1, x2, x3 = x[0], x[1], x[2]
    y1 = 2*x1 + 3*x2 + 4*x3
    return [y1]


def div_eq(x):
    x1, x2 = x[0], x[1]
    y1 = x1/x2
    return [y1]

def div_add(x):
    x1, x2 = x[0], x[1]
    y1 = x1/x2 + 2*x1 - 5*x2
    return [y1]


def mult_eq(x):
    x1, x2 = x[0], x[1]
    y1 = x1*x2
    return [y1]


def pow_eq(x):
    x1, x2 = x[0], x[1]
    y1 = x1**2 + x2**2
    return [y1]


def cube_add_eq(x):
    x1, x2 = x[0], x[1]
    y1 = x1**3 + x2
    return [y1]


def data_gen_de(fn, dt, train_count, x0_train, integrator_keywords):
    train_t = np.arange(0, dt*train_count, dt)
    train_span = (train_t[0], train_t[-1])
    train_x = solve_ivp(fn, train_span, x0_train, t_eval=train_t,
                        **integrator_keywords).y.T
    train_x_dot = np.array([fn(train_t[i], train_x[i]) for i in range(train_t.size)])
    train_x = np.split(train_x, train_x.shape[1], axis=1)
    return train_t, train_x, train_x_dot


def data_gen_eq(fn, x_bounds, x_dim, train_count, validation_split=0.0):
    lower_x, upper_x = x_bounds
    split_x = lower_x + (upper_x - lower_x)*(1 - validation_split)
    lower_x_train = lower_x
    upper_x_train = split_x
    lower_x_test = split_x
    upper_x_test = upper_x

    test_x = np.random.uniform(
        lower_x_test, upper_x_test,
        (x_dim, int(train_count*validation_split))).astype(np.float32)

    train_x = np.random.uniform(
        lower_x_train, upper_x_train,
        (x_dim, train_count)).astype(np.float32)
    print(train_x.shape)
    # train_x = np.split(train_x, x_dim, axis=1)
    # test_x = np.split(test_x, x_dim, axis=1)
    train_y = np.array(fn(train_x)[0])
    test_y = np.array(fn(test_x)[0])
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    # integrator_keywords = {}
    # integrator_keywords['rtol'] = 1e-12
    # integrator_keywords['method'] = 'LSODA'
    # integrator_keywords['atol'] = 1e-12

    # train_t, train_x, train_x_dot = data_gen_de(linear_2D, 0.1, 1000, [1,2],
    #     integrator_keywords)
    
    train_x, train_y, test_x, test_y = data_gen_eq(linear_2D_eq, (1, 10), 2, 1000)
    print(train_x, train_y, test_x, test_y)

