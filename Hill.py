import sympy as sp

def generate_boolecube(tabela, variaveis_str, indice_saida):
    x_bars = sp.symbols(' '.join(variaveis_str))
    if len(variaveis_str) == 1:
        x_bars = (x_bars,)
        
    boolecube = 0
    
    for linha in tabela:
        valores_x = linha[:len(variaveis_str)] 
        B_val = linha[indice_saida]            
        
        if B_val == 1:
            produtorio = 1
            for x_i, x_bar_i in zip(valores_x, x_bars):
                # Se for 1, usa (x). 
                # Se for 0, usa (1 - x).
                termo = x_bar_i if x_i == 1 else (1 - x_bar_i)
                produtorio *= termo
                
            boolecube += produtorio
            
    return sp.simplify(boolecube)

def hillcube_and_normalization(boolecube, variaveis_str, var_saida_str, normalizar_f1=True):

    expr = boolecube
    sufixo_saida = var_saida_str.split('_')[1] if '_' in var_saida_str else var_saida_str[-1]
    
    for v_str in variaveis_str:
        v_sym = sp.Symbol(v_str)
        sufixo_reguladora = v_str.split('_')[1] if '_' in v_str else v_str[-1]
        
        # Define os parâmetros da função de Hill (n) e (k)
        n = sp.Symbol(f'n_{sufixo_reguladora}{sufixo_saida}')
        k = sp.Symbol(f'k_{sufixo_reguladora}{sufixo_saida}')
        N_v = sp.Symbol(f'N({v_str})')
        
        f_v = (N_v**n) / (N_v**n + k**n)
        
        if normalizar_f1:
            f_1 = 1 / (1 + k**n)
            f_v = f_v / f_1
            
        expr = expr.subs(v_sym, f_v)
        
    return sp.simplify(expr)

def make_edo(hillcube_expr, var_saida_str):
    tau = sp.Symbol(f'tau_{var_saida_str}')
    N_saida = sp.Symbol(f'N({var_saida_str})')
    
    edo_expr = (1 / tau) * (hillcube_expr - N_saida)
    
    derivada = sp.Symbol(f'd{var_saida_str}/dt')
    
    return sp.Eq(derivada, edo_expr)

tabela_abc = [
    [0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1]
]

vars_entrada = ['x_a', 'x_b', 'x_c']
mapeamento = {'A': 3, 'B': 4, 'C': 5}

for var, idx in mapeamento.items():
    boole = generate_boolecube(tabela_abc, vars_entrada, idx)
    
    hill = hillcube_and_normalization(boole, vars_entrada, var, normalizar_f1=True)
    
    ode = make_edo(hill, var)
    
    print(f'--- Resultados para {var} ---')
    print(f'BooleCube: {boole}')
    print(f'HillCube (Normalizado): {hill}')
    print(f'ODE: {ode}\n')