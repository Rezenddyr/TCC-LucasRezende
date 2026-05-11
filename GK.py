import numpy as np
import sympy as sp
from scipy.stats import zscore


def gerar_boolecube(tabela, simbolos, coluna_saida):
    n = len(simbolos)
    expressao = sp.Integer(0)
    for linha in tabela:
        entradas = linha[:n]
        saida = linha[coluna_saida]
        if saida == 1:
            produto = sp.Integer(1)
            for val, sym in zip(entradas, simbolos):
                produto *= sym if val == 1 else (1 - sym)
            expressao += produto
    return sp.expand(expressao)


def gerar_gkcube(boolecube, simbolos_entrada, nome_saida):
    expr = boolecube
    for sym in simbolos_entrada:
        reg = sym.name
        N_v = sp.Symbol(f'N_{reg}')
        v2  = sp.Symbol(f'v2_{reg}{nome_saida}')
        J1  = sp.Symbol(f'J1_{reg}{nome_saida}')
        J2  = sp.Symbol(f'J2_{reg}{nome_saida}')

        # Goldbeter-Koshland: fracao ativa no estado estacionario
        # G(v1=N_v, v2, J1, J2) = 2*N_v*J2 / (B + sqrt(B^2 - 4*(v2-N_v)*N_v*J2))
        B    = v2 - N_v + N_v * J2 + v2 * J1
        f_gk = 2 * N_v * J2 / (B + sp.sqrt(B**2 - 4 * (v2 - N_v) * N_v * J2))

        # Normalizacao: G(N_v) / G(1) para que em N_v=1 a funcao valha 1
        # (mesma logica do Hill: f_norm = f_x / f_1)
        B1     = v2 - 1 + J2 + v2 * J1
        f_gk_1 = 2 * J2 / (B1 + sp.sqrt(B1**2 - 4 * (v2 - 1) * J2))
        f_norm = f_gk / f_gk_1

        expr = expr.subs(sym, f_norm)
    return sp.simplify(expr)


def gerar_ode(gkcube, nome_saida):
    tau   = sp.Symbol(f'tau_{nome_saida}')
    N_out = sp.Symbol(f'N_{nome_saida}')
    deriv = sp.Symbol(f'd{nome_saida}_dt')
    return sp.Eq(deriv, (gkcube - N_out) / tau)


def rodar_pipeline(nomes, tabela):
    simbolos = [sp.Symbol(n) for n in nomes]
    N = len(nomes)

    boolecubes = {}
    gkcubes    = {}
    odes       = {}

    for i, var in enumerate(nomes):
        col_saida = N + i

        bc = gerar_boolecube(tabela, simbolos, col_saida)
        boolecubes[var] = bc

        # Passa apenas as variaveis que aparecem no BooleCube
        vars_ativas = [s for s in simbolos if s in bc.free_symbols]
        gkc = gerar_gkcube(bc, vars_ativas, var)
        gkcubes[var] = gkc

        odes[var] = gerar_ode(gkc, var)

    # Coleta todos os parametros livres (v2, J1, J2, tau) para a ES
    simbolos_livres = sorted({
        s for ode in odes.values()
        for s in ode.free_symbols
        if not str(s).startswith('N_') and not str(s).startswith('d')
    }, key=str)

    return boolecubes, gkcubes, odes, simbolos_livres


if __name__ == '__main__':

    tabela = [
        [A, B, C, D, E,  1-E, A, B, C, int((B and D) or (D and E))]
        for A in range(2)
        for B in range(2)
        for C in range(2)
        for D in range(2)
        for E in range(2)
    ]

    boolecubes, gkcubes, odes, livres = rodar_pipeline(
        nomes=['A', 'B', 'C', 'D', 'E'],
        tabela=tabela,
    )

    print("--- BooleCubes ---")
    for var, bc in boolecubes.items():
        print(f"  {var}: {bc}")

    print("\n--- GKCubes ---")
    for var, gkc in gkcubes.items():
        print(f"  {var}: {gkc}")

    print("\n--- ODEs ---")
    for ode in odes.values():
        print(f"  {ode}")

    print("\n--- Simbolos livres para ES ---")
    print(f"  {livres}")
