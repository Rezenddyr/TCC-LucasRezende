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

print("BooleCube A =", boole_A)
print("BooleCube B =", boole_B)
print("BooleCube C =", boole_C)
