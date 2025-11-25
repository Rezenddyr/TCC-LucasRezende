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

def converter_para_hillcube(expr_boole, variaveis):
    expr = expr_boole
    for v in variaveis:
        f_v = f"(({v}**n_{v})/(({v}**n_{v}) + (k_{v}**n_{v})))"
        expr = expr.replace(v, f_v)
        expr = expr.replace(f"(1 - {f_v})", f"(1 - {f_v})")
    return expr


def gerar_ode(expr_hill, var_saida):
    return f"d{var_saida}/dt = ( {expr_hill} - {var_saida} ) / tau_{var_saida}"


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

# === BooleCubes ===
boole_A = gerar_boolecube(tabela, variaveis, 3)
boole_B = gerar_boolecube(tabela, variaveis, 4)
boole_C = gerar_boolecube(tabela, variaveis, 5)

print("BooleCube A =", boole_A)
print("\n")
print("BooleCube B =", boole_B) 
print("\n")
print("BooleCube C =", boole_C)
print("\n")

# === HillCubes ===
hill_A = converter_para_hillcube(boole_A, variaveis)
hill_B = converter_para_hillcube(boole_B, variaveis)
hill_C = converter_para_hillcube(boole_C, variaveis)

print("HillCube A =", hill_A)
print("\n")
print("HillCube B =", hill_B)
print("\n")
print("HillCube C =", hill_C)
print("\n")


# === Gera ODEs ===
ode_A = gerar_ode(hill_A, "x_a")
ode_B = gerar_ode(hill_B, "x_b")
ode_C = gerar_ode(hill_C, "x_c")

print("ODE A:", ode_A)
print("\n")
print("ODE B:", ode_B)
print("\n")
print("ODE C:", ode_C)
print("\n")

