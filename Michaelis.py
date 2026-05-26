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


def gerar_mm(boolecube, simbolos_entrada, nome_saida):
    expr = boolecube

    for sym in simbolos_entrada:
        reg = sym.name
        N_v = sp.Symbol(f'N_{reg}')
        Km  = sp.Symbol(f'Km_{reg}{nome_saida}')
        Vm  = sp.Symbol(f'Vm_{reg}{nome_saida}')

        f_norm = Vm * N_v / (Km + N_v)

        expr = expr.subs(sym, f_norm)

    return sp.simplify(expr)


def gerar_ode(mmcube, nome_saida):
    tau   = sp.Symbol(f'tau_{nome_saida}')
    N_out = sp.Symbol(f'N_{nome_saida}')
    deriv = sp.Symbol(f'd{nome_saida}_dt')
    return sp.Eq(deriv, (mmcube - N_out) / tau)


def rodar_pipeline(nomes, tabela):
    simbolos = [sp.Symbol(n) for n in nomes]
    N = len(nomes)

    boolecubes = {}
    mms        = {}
    odes       = {}

    for i, var in enumerate(nomes):
        col_saida = N + i

        bc = gerar_boolecube(tabela, simbolos, col_saida)
        boolecubes[var] = bc

        vars_ativas = [s for s in simbolos if s in bc.free_symbols]
        mc = gerar_mm(bc, vars_ativas, var)
        mms[var] = mc

        odes[var] = gerar_ode(mc, var)

    simbolos_livres = sorted({
        s for ode in odes.values()
        for s in ode.free_symbols
        if not str(s).startswith('N_') and not str(s).startswith('d')
    }, key=str)

    return boolecubes, mms, odes, simbolos_livres


if __name__ == '__main__':

    tabela = [
        [A, B, C, D, E,  1-E, A, B, C, int((B and D) or (D and E))]
        for A in range(2)
        for B in range(2)
        for C in range(2)
        for D in range(2)
        for E in range(2)
    ]

    boolecubes, mms, odes, livres = rodar_pipeline(
        nomes=['A', 'B', 'C', 'D', 'E'],
        tabela=tabela,
    )

    print("--- BooleCubes ---")
    for var, bc in boolecubes.items():
        print(f"  {var}: {bc}")

    print("\n--- MMs ---")
    for var, mc in mms.items():
        print(f"  {var}: {mc}")

    print("\n--- ODEs ---")
    for ode in odes.values():
        print(f"  {ode}")

    print("\n--- Símbolos livres para ES ---")
    print(f"  {livres}")