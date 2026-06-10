import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import math
import os
import time

METODO = "ABCD_GK"

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

# tau(4) + v2(9) + J1(9) + J2(9)
IND_SIZE     = 31
MIN_STRATEGY = 0.1
MAX_STRATEGY = 10.0

POPULACAO      = []
APTIDAO        = []
APTIDAO_FILHOS = []


def GK(v1, v2, J1, J2):
    if v1 <= 0.0:
        return 0.0
    b    = v2 - v1 + v1 * J2 + v2 * J1
    disc = b * b - 4.0 * (v2 - v1) * v1 * J2
    if disc < 0.0:
        disc = 0.0
    denom = b + math.sqrt(disc)
    if denom <= 1e-12:
        return 1.0
    return 2.0 * v1 * J2 / denom


def twoBody(y, t,
            tauA, tauB, tauC, tauD,
            v2AA, v2AB, v2AD, v2BC, v2BD, v2CA, v2CD, v2DA, v2DD,
            J1AA, J1AB, J1AD, J1BC, J1BD, J1CA, J1CD, J1DA, J1DD,
            J2AA, J2AB, J2AD, J2BC, J2BD, J2CA, J2CD, J2DA, J2DD):

    NA = y[0] / maximo_A
    NB = y[1] / maximo_B
    NC = y[2] / maximo_C
    ND = y[3] / maximo_D

    hAA = GK(NA, v2AA, J1AA, J2AA)
    hAB = GK(NB, v2AB, J1AB, J2AB)
    hAD = GK(ND, v2AD, J1AD, J2AD)
    hBC = GK(NC, v2BC, J1BC, J2BC)
    hBD = GK(ND, v2BD, J1BD, J2BD)
    hCA = GK(NA, v2CA, J1CA, J2CA)
    hCD = GK(ND, v2CD, J1CD, J2CD)
    hDA = GK(NA, v2DA, J1DA, J2DA)
    hDD = GK(ND, v2DD, J1DD, J2DD)

    ydot = np.empty((4,))
    ydot[0] = (((1-hAA)*(1-hAD) + hAB*(1-hAD) + hAA*(1-hAB)*hAD) - NA) / tauA
    ydot[1] = (((1-hBC) + hBD) - NB) / tauB
    ydot[2] = ((hCD + (1-hCA)) - NC) / tauC
    ydot[3] = (((1-hDA)*(1-hDD)) - ND) / tauD
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
        sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind[:IND_SIZE]))
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return float('inf')
        return calcula_diferenca(*organiza_pontos(sol))
    except Exception:
        return float('inf')


def get_bounds(indx):
    if indx < 4:   return 0.1,   5.0    # tau
    if indx < 13:  return 0.01,  1.5    # v2
    return 0.001, 0.99                   # J1, J2


def cria_individuo():
    ind = [r.uniform(*get_bounds(i)) for i in range(IND_SIZE)]
    ind += [r.uniform(MIN_STRATEGY, MAX_STRATEGY) for _ in range(IND_SIZE)]
    return ind


def mutESLogNormal(ind, c, indpb):
    t    = c / math.sqrt(2.0 * math.sqrt(IND_SIZE))
    t0   = c / math.sqrt(2.0 * IND_SIZE)
    t0_n = t0 * r.gauss(0, 1)
    for indx in range(IND_SIZE):
        if r.random() < indpb:
            s_idx = indx + IND_SIZE
            s_old, v_old = ind[s_idx], ind[indx]
            lo, hi = get_bounds(indx)
            tentativas = 0
            while True:
                ind[s_idx] = min(s_old * math.exp(t0_n + t * r.gauss(0, 1)), MAX_STRATEGY)
                ind[indx]  = v_old * ind[s_idx] * r.gauss(0, 1)
                tentativas += 1
                if lo <= ind[indx] <= hi or tentativas > 50:
                    if not (lo <= ind[indx] <= hi):
                        ind[s_idx], ind[indx] = s_old, v_old
                    break
    return ind


def varOr(populacao, lambda_):
    offspring = []
    for _ in range(lambda_):
        filho = mutESLogNormal(copy.deepcopy(r.choice(populacao)), 1.0, 0.03)
        offspring.append(filho)
    return offspring


def selTournament(offspring, mu, tournsize):
    indices = list(range(len(APTIDAO_FILHOS)))
    chosen, chosen_apt = [], []
    for _ in range(mu):
        asp   = r.sample(indices, min(tournsize, len(indices)))
        apts  = [APTIDAO_FILHOS[i] for i in asp]
        menor = min(apts)
        chosen.append(offspring[asp[apts.index(menor)]])
        chosen_apt.append(menor)
    return chosen, chosen_apt


def plota_resultados(ind, pasta, seed):
    sol = odeint(twoBody, Y0, dobra_pontos, args=tuple(ind[:IND_SIZE]))
    pA, pB, pC, pD = organiza_pontos(sol)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Resultados — {METODO}', fontsize=14)
    for idx, (pred, orig, nome) in enumerate([(pA, A_ORIGINAL, 'A'), (pB, B_ORIGINAL, 'B'),
                                               (pC, C_ORIGINAL, 'C'), (pD, D_ORIGINAL, 'D')]):
        ax = axs[idx // 2][idx % 2]
        ax.plot(x, pred, label=f'{nome} predito')
        ax.plot(x, orig,  label=f'{nome} real')
        ax.set_title(f'Variável {nome}'); ax.set_xlabel('Tempo')
        ax.set_ylabel('Concentração');    ax.legend()
    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300); plt.close()
    print(f"Gráfico salvo em: {caminho}")


def main(seed):
    POPULACAO.clear(); APTIDAO.clear()
    r.seed(seed)
    MU, LAMBDA = 15, 105
    pasta = METODO; os.makedirs(pasta, exist_ok=True)
    arq_res = os.path.join(pasta, f'resultados_{METODO}_seed{seed}.txt')
    inicio  = time.time()
    with open(arq_res, "w") as f:
        f.write(f"SEED: {seed}\nMETODO: {METODO}\nIND_SIZE: {IND_SIZE}\n")

    for _ in range(MU): POPULACAO.append(cria_individuo())
    for ind in POPULACAO: APTIDAO.append(avalia(ind))

    for gen in range(10001):
        offspring = varOr(POPULACAO, LAMBDA)
        APTIDAO_FILHOS.clear()
        for ind in offspring:
            dif = avalia(ind)
            if math.isfinite(dif): APTIDAO_FILHOS.append(dif)
        novos_pais, novas_aptidoes = selTournament(offspring, MU, 3)
        POPULACAO.clear(); POPULACAO.extend(novos_pais)
        APTIDAO.clear();   APTIDAO.extend(novas_aptidoes)
        if gen % 100 == 0:
            mv = min(APTIDAO); mi = APTIDAO.index(mv)
            print(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi][:IND_SIZE]}\nAPTIDAO: \n{mv}")
            with open(arq_res, "a") as f:
                f.write(f"GEN: {gen}\nIndividuo: \n{POPULACAO[mi][:IND_SIZE]}\nAPTIDAO: \n{mv}\n")

    mv = min(APTIDAO); mi = APTIDAO.index(mv)
    tt = time.time() - inicio
    msg = (f"\n--- RESULTADO FINAL ---\nErro (norma-1): {mv}\n"
           f"Parâmetros: {POPULACAO[mi][:IND_SIZE]}\n"
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
