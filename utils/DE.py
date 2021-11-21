import numpy as np
import random as rand
import matplotlib; matplotlib.use('module://backend_interagg')
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math


def main():

    Np = 100    # Number of individuals
    D = 2       # Dimension
    Cr = 0.9    # Crossover probability
    F = 0.5     # Mutation factor

    VectorV = np.empty((Np, D))
    VectorU = np.empty((Np, D))

    # Initialize the arrays
    for i in range(Np):
        for j in range(2):
            VectorV[i][j] = rand.randint(-20, 20)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)


    NumEvaluations = 0  # Maximum number of iterations
    while (NumEvaluations < 100):
        for i in range(Np):
            r0 = i
            while (r0 == i):
                r0 = math.floor(rand.random() * Np)
            r1 = r0
            while (r1 == r0 or r1 == i):
                r1 = math.floor(rand.random() * Np)
            r2 = r1
            while (r2 == r1 or r2 == r0 or r2 == i):
                r2 = math.floor(rand.random() * Np)

            jrand = math.floor(rand.random() * D)

            for j in range(D):
                if (rand.random() <= Cr or j == jrand):
                    # Mutation
                    VectorU[i][j] = VectorV[r0][j] + F * (VectorV[r1][j] - VectorV[r2][j])
                else:
                    VectorU[i][j] = VectorV[i][j]

        for k in range(Np):
            if fitness(VectorU[k][0], VectorU[k][1]) < fitness(VectorV[k][0], VectorV[k][1]):
                VectorV[k][0] = VectorU[k][0]
                VectorV[k][1] = VectorU[k][1]

        line1 = ax.plot(VectorU[:, 0], VectorU[:, 1], 'b+')
        line2 = ax.plot(VectorV[:, 0], VectorV[:, 1], 'g*')

        ax.set_xlim(-10, 20)
        ax.set_ylim(-10, 20)

        fig.canvas.draw()

        ax.clear()
        ax.grid(True)

        NumEvaluations += 1

        print('number of evaluations: ' + str(NumEvaluations))

    print('VectorV: ')
    print(VectorV)

    plt.show()


def fitness(x, y):
    # TO DO: implement our own fitness function. Input: the parameters of the controller. Example below, Function Rosenbrock.
    return 100 * ((y - (x ** 2)) ** 2) + ((1 - (x ** 2)) ** 2)


if '__main__' == main():
    main()
