from sympy import symbols, powsimp, ln, preorder_traversal, Float, sqrt, count_ops


def get_sympy_expr(model_weights, x_dim):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('X_'+str(i+1), positive=True))
    sym_expr = (powsimp(sym[0]**model_weights[2][0][0])*powsimp(sym[1]**model_weights[2][1][0]))*model_weights[5][0][0] + \
               (sym[0]*model_weights[3][0][0] + sym[1]*model_weights[3][1][0])*model_weights[5][1][0]

    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            sym_expr = sym_expr.subs(a, round(a, 2))
    print(sym_expr)


def round_sympy_expr(sym_expr, round_digits=2, norm_thresh=0.01):
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            if int(round(a, round_digits)) == round(a, round_digits):
                sym_expr = sym_expr.subs(a, int(round(a, round_digits)))
            else:
                sym_expr = sym_expr.subs(a, round(a, round_digits))
    return sym_expr


def is_sqrt_term(expr):
    if count_ops(expr) != 1 or len(expr.free_symbols) != 1:
        return None
    ops = list(count_ops(expr, visual=True).free_symbols)
    op = ops[0].name
    op_sym = list(expr.free_symbols)[0]
    if op == 'POW' and op_sym**0.5 == expr:
        return sqrt(op_sym)
    elif op == 'POW' and op_sym**(-0.5) == expr:
        return 1/sqrt(op_sym)
    else:
        return None

def get_sympy_expr_v2(model, x_dim, ln_layer_count=2, round_digits=3,
                      norm_thresh=0.01, x_cols=None):
    if x_cols:
        sym = x_cols
    else:
        sym = []
        for i in range(x_dim):
            sym.append(symbols('X_'+str(i), positive=True))

    ln_dense_weights = [model.get_layer('ln_dense_{}'.format(i)).get_weights() for i in range(ln_layer_count)]
    output_dense_weights = model.get_layer('output_dense').get_weights()
    print(ln_dense_weights)
    print(output_dense_weights)
    sym_expr = 0
    for i in range(ln_layer_count):
        ln_block_expr = output_dense_weights[0][i][0]
        for j in range(x_dim):
            ln_block_expr *= sym[j]**ln_dense_weights[i][0][j][0]
        sym_expr += ln_block_expr
    sym_expr = round_sympy_expr(sym_expr, round_digits=round_digits,
                                norm_thresh=norm_thresh)

    print(sym_expr)
    
    return sym_expr
