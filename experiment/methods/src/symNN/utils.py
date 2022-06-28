from sympy import symbols, powsimp, ln, preorder_traversal, Float


def get_sympy_expr(model_weights, x_dim):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('x'+str(i+1)))
    sym_expr = (powsimp(sym[0]**model_weights[2][0][0])*powsimp(sym[1]**model_weights[2][1][0]))*model_weights[5][0][0] + \
               (sym[0]*model_weights[3][0][0] + sym[1]*model_weights[3][1][0])*model_weights[5][1][0]

    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            sym_expr = sym_expr.subs(a, round(a, 2))
    print(sym_expr)


def round_sympy_expr(sym_expr, round_digits=2, norm_thresh=0.01):
    sum_coeff = 0
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            sum_coeff += a
    for a in preorder_traversal(sym_expr):
        if isinstance(a, Float):
            if a/sum_coeff < norm_thresh:
                sym_expr = sym_expr.subs(a, 0)
            sym_expr = sym_expr.subs(a, round(a, 2))
    return sym_expr

def get_sympy_expr_v2(model, x_dim, ln_layer_count=2, round_digits=3,
                      norm_thresh=0.01):
    sym = []
    for i in range(x_dim):
        sym.append(symbols('x'+str(i+1)))
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
