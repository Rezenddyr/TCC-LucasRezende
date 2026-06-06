import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import os
import time

METODO = "DE_GRN10_Hill"

# ---------------------------------------------------------------------------
# Rede GRN10 - 10 genes (A..J) - cinetica de Hill
#
# Topologia (regulador -> alvo):
#   A <- J (repressao)         B <- E
#   C <- B, F, A (logico)      D <- F
#   E <- J (repressao)         F <- A
#   G <- B, F, A (logico)      H <- F
#   I <- G E H (produto)       J <- I
# ---------------------------------------------------------------------------

arquivo = open("GRN10.txt", 'r')
x = []
cols = [[] for _ in range(10)]
for linha in arquivo:
    e = linha.split()
    if len(e) < 11:
        continue
    x.append(float(e[0]))
    for i in range(10):
        cols[i].append(float(e[i + 1]))
arquivo.close()

ORIG = [copy.deepcopy(c) for c in cols]
MAXS = np.array([max(c) for c in cols])
dobra_pontos = copy.deepcopy(x)
Y0 = [c[0] for c in cols]

NOMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

AJ, BE, CB, CF, CA, DF, EJ, FA, GB, GF, GA, HF, IG, IH, JI = range(15)

# tau(10) + k(15) + n(15) + vmax(10) = 50
IND_SIZE  = 50
TAU_SIZE  = 10
K_SIZE    = 15
N_SIZE    = 15
VMAX_SIZE = 10

MIN_TAU  = 0.1
MAX_TAU  = 5.0
MIN_K    = 0.1
MAX_K    = 1.0
MIN_N    = 1.0
MAX_N    = 25.0
MIN_VMAX = 0.1
MAX_VMAX = 2.0

BOUNDS = (
    [(MIN_TAU,  MAX_TAU)]  * TAU_SIZE +
    [(MIN_K,    MAX_K)]    * K_SIZE +
    [(MIN_N,    MAX_N)]    * N_SIZE +
    [(MIN_VMAX, MAX_VMAX)] * VMAX_SIZE
)


def sum6(a, b, c):
    return (a * (1 - b) * (1 - c) + (1 - a) * b * (1 - c) +
            (1 - a) * (1 - b) * c + a * (1 - b) * c +
            (1 - a) * b * c + a * b * c)


def twoBody(y, t, p):
    tau  = p[0:10]
    k    = p[10:25]
    n    = p[25:40]
    vmax = p[40:50]

    N = y / MAXS

    def H(e, val):
        if val <= 0.0:
            return 0.0
        ne = int(n[e])
        ve = pow(val, ne)
        return ve / (ve + pow(k[e], ne))

    ydot = np.empty((10,))
    ydot[0] = (vmax[0] * (1.0 - H(AJ, N[9])) - N[0]) / tau[0]                              # A <- J (rep)
    ydot[1] = (vmax[1] * H(BE, N[4]) - N[1]) / tau[1]                                       # B <- E
    ydot[2] = (vmax[2] * sum6(H(CB, N[1]), H(CF, N[5]), H(CA, N[0])) - N[2]) / tau[2]       # C <- B,F,A
    ydot[3] = (vmax[3] * H(DF, N[5]) - N[3]) / tau[3]                                       # D <- F
    ydot[4] = (vmax[4] * (1.0 - H(EJ, N[9])) - N[4]) / tau[4]                              # E <- J (rep)
    ydot[5] = (vmax[5] * H(FA, N[0]) - N[5]) / tau[5]                                       # F <- A
    ydot[6] = (vmax[6] * sum6(H(GB, N[1]), H(GF, N[5]), H(GA, N[0])) - N[6]) / tau[6]       # G <- B,F,A
    ydot[7] = (vmax[7] * H(HF, N[5]) - N[7]) / tau[7]                                       # H <- F
    ydot[8] = (vmax[8] * H(IG, N[6]) * H(IH, N[7]) - N[8]) / tau[8]                         # I <- G e H
    ydot[9] = (vmax[9] * H(JI, N[8]) - N[9]) / tau[9]                                       # J <- I
    return ydot


def organiza_pontos(sol):
    return [list(sol[:, i]) for i in range(10)]


def calcula_diferenca(preds):
    dif = 0.0
    for i in range(10):
        oi, pi = ORIG[i], preds[i]
        for j in range(len(pi)):
            dif += abs(oi[j] - pi[j])
    return dif


def avalia(ind):
    try:
        sol = odeint(twoBody, Y0, dobra_pontos,
                     args=(np.asarray(ind[:IND_SIZE]),))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        return calcula_diferenca(organiza_pontos(sol))
    except Exception:
        return float('inf')


def plota_resultados(ind, pasta, seed):
    sol = odeint(twoBody, Y0, dobra_pontos, args=(np.asarray(ind[:IND_SIZE]),))
    preds = organiza_pontos(sol)

    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f'Resultados - {METODO}', fontsize=14)
    for idx in range(10):
        ax = axs[idx // 5][idx % 5]
        ax.plot(x, preds[idx], label=f'{NOMES[idx]} predito')
        ax.plot(x, ORIG[idx],  label=f'{NOMES[idx]} real')
        ax.set_title(f'Variavel {NOMES[idx]}')
        ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Concentracao')
        ax.legend()

    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Grafico salvo em: {caminho}")


def main(seed):
    pasta = METODO
    os.makedirs(pasta, exist_ok=True)

    arquivo_resultados = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')

    inicio = time.time()
    with open(arquivo_resultados, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"strategy=best1bin  popsize=15  F=0.8  CR=0.75  polish=True\n")
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  K=[{MIN_K}, {MAX_K}]"
                f"  N=[{MIN_N}, {MAX_N}]  Vmax=[{MIN_VMAX}, {MAX_VMAX}]\n")

    gen_log = [0]

    def callback(xk, convergence=None):
        if gen_log[0] % 100 == 0:
            apt = avalia(xk)
            print(f"GEN: {gen_log[0]}")
            print(f"Individuo: \n{list(xk)}")
            print(f"APTIDAO: \n{apt}")
            with open(arquivo_resultados, "a") as f:
                f.write(f"GEN: {gen_log[0]}\nIndividuo: \n{list(xk)}\nAPTIDAO: \n{apt}\n")
        gen_log[0] += 1

    result = differential_evolution(
        avalia,
        BOUNDS,
        strategy='best1bin',
        popsize=15,
        mutation=0.8,
        recombination=0.75,
        polish=True,
        seed=seed,
        callback=callback,
        maxiter=10001,
        tol=0,
        updating='immediate'
    )

    melhor_ind  = [float(v) for v in result.x]
    menor_valor = result.fun

    tempo_total = time.time() - inicio
    horas    = int(tempo_total // 3600)
    minutos  = int((tempo_total % 3600) // 60)
    segundos = int(tempo_total % 60)

    print(f"\n--- RESULTADO FINAL ---")
    print(f"Erro (norma-1): {menor_valor}")
    print(f"Parametros: {melhor_ind}")
    print(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")

    with open(arquivo_resultados, "a") as f:
        f.write(f"\n--- RESULTADO FINAL ---\n")
        f.write(f"Erro (norma-1): {menor_valor}\n")
        f.write(f"Parametros: {melhor_ind}\n")
        f.write(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s\n")

    plota_resultados(melhor_ind, pasta, seed)


SEEDS = [
    1778434285, 1778461231, 1778490666, 1778578247, 1778663936,
    1778719796, 1778749666, 1778837425, 1778893788, 1778981565,
]

if __name__ == "__main__":
    for s in SEEDS:
        main(s)
