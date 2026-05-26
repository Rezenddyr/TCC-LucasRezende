import matplotlib.pyplot as plt
import copy
import random as r
import numpy as np
from scipy.integrate import odeint
import math
import os
import time

METODO = "ES_MM"

arquivo = open("GRN5.txt", 'r')
x, A, B, C, D, E = [], [], [], [], [], []
for linha in arquivo:
    elementos = linha.split()
    x.append(float(elementos[0].strip()))
    A.append(float(elementos[1].strip()))
    B.append(float(elementos[2].strip()))
    C.append(float(elementos[3].strip()))
    D.append(float(elementos[4].strip()))
    E.append(float(elementos[5].strip()))
arquivo.close()

maximo_A = max(A)
maximo_B = max(B)
maximo_C = max(C)
maximo_D = max(D)
maximo_E = max(E)

A_ORIGINAL = copy.deepcopy(A)
B_ORIGINAL = copy.deepcopy(B)
C_ORIGINAL = copy.deepcopy(C)
D_ORIGINAL = copy.deepcopy(D)
E_ORIGINAL = copy.deepcopy(E)

dobra_pontos = copy.deepcopy(x)
Y0 = [A[0], B[0], C[0], D[0], E[0]]

# tau(5) + k(7) + Vmax(5) = 17
IND_SIZE     = 17
MIN_STRATEGY = 0.1
MAX_STRATEGY = 10.0

POPULACAO      = []
APTIDAO        = []
APTIDAO_FILHOS = []


def mm(v, k):
    if v <= 0.0:
        return 0.0
    return v / (v + k)


def twoBody(y, t,
            tauA, tauB, tauC, tauD, tauE,
            kA, kB, kC, kD, kEB, kED, kEE,
            vmaxA, vmaxB, vmaxC, vmaxD, vmaxE):

    ydot = np.empty((5,))

    ydot[0] = (vmaxA * (1 - mm(y[4]/maximo_E, kA)) - y[0]/maximo_A) / tauA
    ydot[1] = (vmaxB * mm(y[0]/maximo_A, kB)       - y[1]/maximo_B) / tauB
    ydot[2] = (vmaxC * mm(y[1]/maximo_B, kC)       - y[2]/maximo_C) / tauC
    ydot[3] = (vmaxD * mm(y[2]/maximo_C, kD)       - y[3]/maximo_D) / tauD

    HB = mm(y[1]/maximo_B, kEB)
    HD = mm(y[3]/maximo_D, kED)
    HE = mm(y[4]/maximo_E, kEE)
    ydot[4] = (vmaxE * ((HB * HD) + (HD * HE)) - y[4]/maximo_E) / tauE

    return ydot


def organiza_pontos(sol):
    pA, pB, pC, pD, pE = [], [], [], [], []
    for pt in sol:
        pA.append(pt[0]); pB.append(pt[1]); pC.append(pt[2])
        pD.append(pt[3]); pE.append(pt[4])
    return pA, pB, pC, pD, pE


def calcula_diferenca(pA, pB, pC, pD, pE):
    dif = 0
    for i in range(len(pA)):
        dif += abs(A_ORIGINAL[i] - pA[i])
        dif += abs(B_ORIGINAL[i] - pB[i])
        dif += abs(C_ORIGINAL[i] - pC[i])
        dif += abs(D_ORIGINAL[i] - pD[i])
        dif += abs(E_ORIGINAL[i] - pE[i])
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
    if indx < 5:   return 0.1,   5.0
    if indx < 12:  return 0.001, 0.999
    return 1.0, 10.0


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
    pA, pB, pC, pD, pE = organiza_pontos(sol)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Resultados — {METODO}', fontsize=14)
    dados = [(pA, A_ORIGINAL, 'A'), (pB, B_ORIGINAL, 'B'),
             (pC, C_ORIGINAL, 'C'), (pD, D_ORIGINAL, 'D'),
             (pE, E_ORIGINAL, 'E')]
    for idx, (pred, orig, nome) in enumerate(dados):
        ax = axs[idx // 3][idx % 3]
        ax.plot(x, pred, label=f'{nome} predito')
        ax.plot(x, orig,  label=f'{nome} real')
        ax.set_title(f'Variável {nome}')
        ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Concentração')
        ax.legend()
    axs[1][2].axis('off')
    plt.tight_layout()
    caminho = os.path.join(pasta, f'graficos_{METODO}_seed{seed}.png')
    plt.savefig(caminho, dpi=300); plt.close()
    print(f"Gráfico salvo em: {caminho}")


def main(seed):
    POPULACAO.clear(); APTIDAO.clear(); APTIDAO_FILHOS.clear()
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
