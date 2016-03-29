import matplotlib.pyplot as plt

from math import exp, sqrt, pi
from numpy import average, array_split


def pdf(x, theta, sigma=1):
    return exp(-0.5 * ((x - theta) ** 2) / (sigma ** 2)) / (sigma * sqrt(2 * pi))


def e(xs, thetas, i, j):
    return pdf(xs[i], thetas[j]) / sum(pdf(xs[i], thetas[n]) for n in range(len(thetas)))


def m(xs, es, j):
    return sum(es[j][i] * xs[i] for i in range(len(xs))) / sum(es[j])


def plot(initial_data, thetas):
    for theta in thetas:
        plt.scatter([theta], [1], s=[100], color='#FF0000')

    plt.scatter(initial_data, [1] * len(initial_data), linestyle='dotted')

    plt.show()


def main(number_of_arrays=2, number_of_iterations=1000):
    xs = [float(line.strip()) for line in open('data/sample-data.txt').readlines()]
    thetas = [average(array) for array in array_split(sorted(xs), number_of_arrays)]

    for i in range(number_of_iterations):
        es = [[e(xs, thetas, i, j) for i in range(len(xs))] for j in range(len(thetas))]
        ms = [m(xs, es, j) for j in range(len(thetas))]

        if ms == thetas:
            print('\nFinished. Total number of iterations: %d\n' % (i + 1))

            break

        thetas = ms

        if i in [4, 9]:
            print('Iteration %d:\t%s' % (i + 1, str(thetas)))

    print('Final results:\t%s' % str(thetas))

    plot(xs, thetas)


main()
