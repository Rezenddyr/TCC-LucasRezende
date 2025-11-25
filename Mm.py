import sympy as sp

def gerar_boolecube(tabela, variaveis, indice_saida):
    x = sp.symbols(variaveis)
    termos = []
    for linha in tabela:
        entradas = linha[:len(variaveis)]
        saida = linha[indice_saida]
        if saida == 1:
            termo_partes = []
            for xi, xi_val in zip(x, entradas):
                if xi_val == 1:
                    termo_partes.append(f"{xi}")
                else:
                    termo_partes.append(f"(1 - {xi})")
            termos.append(" * ".join(termo_partes))
    expressao = " + ".join(termos)
    return expressao

def converter_para_michaelis(expr_boole, variaveis):
    expr = expr_boole
    for v in variaveis:
        f_v = f"({v}/({v} + K_{v}))"
        expr = expr.replace(v, f_v)
        expr = expr.replace(f"(1 - {f_v})", f"(1 - {f_v})")
    return expr

def gerar_ode(expr_mm, var_saida):
    return f"d{var_saida}/dt = ( {expr_mm} - {var_saida} ) / tau_{var_saida}"

tabela = [
    [0,0,0,0,1,0],
    [0,0,1,1,1,0],
    [0,1,0,1,0,1],
    [0,1,1,0,1,1],
    [1,0,0,1,1,0],
    [1,0,1,0,1,1],
    [1,1,0,0,0,1],
    [1,1,1,1,0,1],
]

variaveis = ['x_a', 'x_b', 'x_c']

boole_A = gerar_boolecube(tabela, variaveis, 3)
boole_B = gerar_boolecube(tabela, variaveis, 4)
boole_C = gerar_boolecube(tabela, variaveis, 5)

print("BooleCube A =", boole_A, "\n")
print("BooleCube B =", boole_B, "\n")
print("BooleCube C =", boole_C, "\n")

mm_A = converter_para_michaelis(boole_A, variaveis)
mm_B = converter_para_michaelis(boole_B, variaveis)
mm_C = converter_para_michaelis(boole_C, variaveis)

print("MM Cube A =", mm_A, "\n")
print("MM Cube B =", mm_B, "\n")
print("MM Cube C =", mm_C, "\n")

ode_A = gerar_ode(mm_A, "x_a")
ode_B = gerar_ode(mm_B, "x_b")
ode_C = gerar_ode(mm_C, "x_c")

print("ODE A:", ode_A, "\n")
print("ODE B:", ode_B, "\n")
print("ODE C:", ode_C, "\n")
