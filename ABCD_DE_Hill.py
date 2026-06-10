import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import os
import time

METODO = "ABCD_DE_Hill"

arquivo = open("Dados_abcd.txt", 'r')
x, A, B, C, D = [], [], [], [], []
primeira = True
for linha in arquivo:
    if primeira:
        primeira = False
        continue
    el = linha.split()
    if not el:
        continue
    x.append(float(el[0]))
    A.append(float(el[1]))
    B.append(float(el[2]))
    C.append(float(el[3]))
    D.append(float(el[4]))
arquivo.close()

maximo_A = max(A)
maximo_B = max(B)
maximo_C = max(C)
maximo_D = max(D)

A_ORIGINAL = copy.deepcopy(A)
B_ORIGINAL = copy.deepcopy(B)
C_ORIGINAL = copy.deepcopy(C)
D_ORIGINAL = copy.deepcopy(D)

dobra_pontos = copy.deepcopy(x)
Y0 = [A[0], B[0], C[0], D[0]]

# tau(4) + k(4) + n(4) + Vmax(3) = 15
MIN_TAU  = 0.1
MAX_TAU  = 5.0
MIN_K    = 0.001
MAX_K    = 0.999
MIN_N    = 1.0
MAX_N    = 10.0
MIN_VMAX = 1.0
MAX_VMAX = 10.0

BOUNDS = (
    [(MIN_TAU,  MAX_TAU)]  * 4 +
    [(MIN_K,    MAX_K)]    * 4 +
    [(MIN_N,    MAX_N)]    * 4 +
    [(MIN_VMAX, MAX_VMAX)] * 3
)


def hill(v, n, k):
    if v <= 0.0:
        return 0.0
    vn = v ** n
    return vn / (vn + k ** n)


def twoBody(y, t,
            tauA, tauB, tauC, tauD,
            kBA, kBD, kCB, kDC,
            nBA, nBD, nCB, nDC,
            VmaxB, VmaxC, VmaxD):

    NA = y[0] / maximo_A
    NB = y[1] / maximo_B
    NC = y[2] / maximo_C
    ND = y[3] / maximo_D

    hBA = hill(NA, nBA, kBA)
    hBD = hill(ND, nBD, kBD)
    hCB = hill(NB, nCB, kCB)
    hDC = hill(NC, nDC, kDC)

    ydot = np.empty((4,))
    ydot[0] = (1 - NA) / tauA
    ydot[1] = (VmaxB * hBA * hBD - NB) / tauB
    ydot[2] = (VmaxC * hCB - NC) / tauC
    ydot[3] = (VmaxD * (1 - hDC) - ND) / tauD
    return ydot


def organiza_pontos(sol):
    pA, pB, pC, pD = [], [], [], []
    for pt in sol:
        pA.append(pt[0]); pB.append(pt[1])
        pC.append(pt[2]); pD.append(pt[3])
    return pA, pB, pC, pD


def calcula_diferenca(pA, pB, pC, pD):
    dif = 0
    for i in range(len(pA)):
        dif += abs(A_ORIGINAL[i] - pA[i])
        dif += abs(B_ORIGINAL[i] - pB[i])
        dif += abs(C_ORIGINAL[i] - pC[i])
        dif += abs(D_ORIGINAL[i] - pD[i])
    return dif


def avalia(ind):
    try:
        sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        return calcula_diferenca(*organiza_pontos(sol))
    except Exception:
        return float('inf')


def plota_resultados(ind, pasta, seed):
    sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind))
    pA, pB, pC, pD = organiza_pontos(sol)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Resultados — {METODO}', fontsize=14)
    for idx, (pred, orig, nome) in enumerate([(pA, A_ORIGINAL, 'A'), (pB, B_ORIGINAL, 'B'),
                                               (pC, C_ORIGINAL, 'C'), (pD, D_ORIGINAL, 'D')]):
        ax = axs[idx // 2][idx % 2]
        ax.plot(x, pred, label=f'{nome} predito')
        ax.plot(x, orig,  label=f'{nome} real')
        ax.set_title(f'Variável {nome}'); ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Concentração');    ax.legend()
    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300); plt.close()
    print(f"Gráfico salvo em: {caminho}")


def main(seed):
    pasta = METODO
    os.makedirs(pasta, exist_ok=True)

    arquivo_resultados = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')

    inicio = time.time()
    with open(arquivo_resultados, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"strategy=best1bin  popsize=15  F=0.8  CR=0.75  polish=True\n")
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  k=[{MIN_K}, {MAX_K}]"
                f"  n=[{MIN_N}, {MAX_N}]  Vmax=[{MIN_VMAX}, {MAX_VMAX}]\n")

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
    print(f"Parâmetros: {melhor_ind}")
    print(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")

    with open(arquivo_resultados, "a") as f:
        f.write(f"\n--- RESULTADO FINAL ---\n")
        f.write(f"Erro (norma-1): {menor_valor}\n")
        f.write(f"Parâmetros: {melhor_ind}\n")
        f.write(f"Tempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s\n")

    plota_resultados(melhor_ind, pasta, seed)


SEEDS = [
    1778434285, 1778461231, 1778490666, 1778578247, 1778663936,
    1778719796, 1778749666, 1778837425, 1778893788, 1778981565,
]

if __name__ == "__main__":
    for s in SEEDS:
        main(s)
