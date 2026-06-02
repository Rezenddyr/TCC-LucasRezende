import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import os
import time

METODO = "DE_ABCD_MM"

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

# tau(4) + k(4) + Vmax(3) = 11
IND_SIZE  = 11
TAU_SIZE  = 4
K_SIZE    = 4
VMAX_SIZE = 3

MIN_TAU  = 0.1
MAX_TAU  = 5.0
MIN_K    = 0.001
MAX_K    = 0.999
MIN_VMAX = 1.0
MAX_VMAX = 10.0

F  = 0.8
CR = 0.75

POPULACAO = []
APTIDAO   = []


def mm(v, k):
    if v <= 0.0:
        return 0.0
    return v / (v + k)


def twoBody(y, t,
            tauA, tauB, tauC, tauD,
            kBA, kBD, kCB, kDC,
            VmaxB, VmaxC, VmaxD):

    NA = y[0] / maximo_A
    NB = y[1] / maximo_B
    NC = y[2] / maximo_C
    ND = y[3] / maximo_D

    mBA = mm(NA, kBA)
    mBD = mm(ND, kBD)
    mCB = mm(NB, kCB)
    mDC = mm(NC, kDC)

    ydot = np.empty((4,))
    ydot[0] = (1 - NA) / tauA
    ydot[1] = (VmaxB * mBA * mBD - NB) / tauB
    ydot[2] = (VmaxC * mCB - NC) / tauC
    ydot[3] = (VmaxD * (1 - mDC) - ND) / tauD
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


def cria_individuo():
    ind = []
    for _ in range(TAU_SIZE):  ind.append(r.uniform(MIN_TAU,  MAX_TAU))
    for _ in range(K_SIZE):    ind.append(r.uniform(MIN_K,    MAX_K))
    for _ in range(VMAX_SIZE): ind.append(r.uniform(MIN_VMAX, MAX_VMAX))
    return ind


def limites_gene(j):
    if j < TAU_SIZE:
        return MIN_TAU, MAX_TAU
    elif j < TAU_SIZE + K_SIZE:
        return MIN_K, MAX_K
    else:
        return MIN_VMAX, MAX_VMAX


def mutacao_DE(populacao, aptidao, idx):
    melhor_idx = aptidao.index(min(aptidao))
    candidatos = [i for i in range(len(populacao)) if i != idx]
    r1, r2 = r.sample(candidatos, 2)
    mutante = []
    for j in range(IND_SIZE):
        m = populacao[melhor_idx][j] + F * (populacao[r1][j] - populacao[r2][j])
        mutante.append(m)
    return mutante


def cruzamento_DE(alvo, mutante):
    j_rand = r.randint(0, IND_SIZE - 1)
    trial = []
    for j in range(IND_SIZE):
        lo, hi = limites_gene(j)
        if r.random() < CR or j == j_rand:
            trial.append(max(lo, min(hi, mutante[j])))
        else:
            trial.append(alvo[j])
    return trial


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
    POPULACAO.clear(); APTIDAO.clear()
    r.seed(seed)
    MU = 15 * IND_SIZE
    pasta = METODO; os.makedirs(pasta, exist_ok=True)
    arq_res = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')
    inicio  = time.time()
    with open(arq_res, "w") as f:
        f.write(f"SEED: {seed}\nMETODO: {METODO}\nIND_SIZE: {IND_SIZE}\n")
        f.write(f"strategy=best1bin  popsize={MU}  F={F}  CR={CR}\n")
        f.write(f"BOUNDS: tau=[{MIN_TAU}, {MAX_TAU}]  k=[{MIN_K}, {MAX_K}]  Vmax=[{MIN_VMAX}, {MAX_VMAX}]\n")

    for _ in range(MU): POPULACAO.append(cria_individuo())
    for ind in POPULACAO: APTIDAO.append(avalia(ind))

    for gen in range(10001):
        for i in range(MU):
            mutante = mutacao_DE(POPULACAO, APTIDAO, i)
            trial   = cruzamento_DE(POPULACAO[i], mutante)
            apt_trial = avalia(trial)
            if apt_trial <= APTIDAO[i]:
                POPULACAO[i] = trial
                APTIDAO[i]   = apt_trial

        if gen % 100 == 0:
            mv = min(APTIDAO); mi = APTIDAO.index(mv)
            print(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi]}\nAPTIDAO: \n{mv}")
            with open(arq_res, "a") as f:
                f.write(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi]}\nAPTIDAO: \n{mv}\n")

    mv = min(APTIDAO); mi = APTIDAO.index(mv)
    tt = time.time() - inicio
    msg = (f"\n--- RESULTADO FINAL ---\nErro (norma-1): {mv}\n"
           f"Parâmetros: {POPULACAO[mi]}\n"
           f"Tempo total: {int(tt//3600):02d}h {int((tt%3600)//60):02d}m {int(tt%60):02d}s\n")
    print(msg)
    with open(arq_res, "a") as f: f.write(msg)
    plota_resultados(POPULACAO[mi], pasta, seed)


SEEDS = [
    1778434285, 1778461231, 1778490666, 1778578247, 1778663936,
    1778719796, 1778749666, 1778837425, 1778893788, 1778981565,
]

if __name__ == "__main__":
    for s in SEEDS:
        main(s)
