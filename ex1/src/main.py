from random import uniform
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate, p=2, threshold=0.01):
        self.p = p
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.N = 0
        self.W = self.random_weights()
        self.b = self.random_value()
        self.training_features = []

    @staticmethod
    def random_value(min_value=-1.0, max_value=1.0):
        return uniform(min_value, max_value)

    def random_weights(self):
        return [self.random_value() for _ in range(self.p)]

    def f(self, feature):
        return self.b + sum(self.W[i] * feature[i] for i in range(self.p))

    def squared_error(self, feature):
        return (self.f(feature[0]) - feature[1]) ** 2

    def l(self, features=None):
        if features is None:
            features = self.training_features

        return (1.0/self.N) * sum(self.squared_error(feature) for feature in features)

    def l_gradient_w(self, i):
        return (2.0/self.N) * sum((self.f(feature[0]) - feature[1]) * feature[0][i]
                                  for feature in self.training_features)

    def l_gradient_b(self):
        return (2.0/self.N) * sum((self.f(feature[0]) - feature[1]) for feature in self.training_features)

    @staticmethod
    def load_features(train=True):
        features = []

        with open('data/data-%s.csv' % ('train' if train else 'test'), 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split(',')))

                features.append([values[:2], values[-1]])

        return features

    def update_weights(self):
        for i in range(self.p):
            self.W[i] -= self.l_gradient_w(i)

    def update_bias(self):
        self.b -= self.learning_rate * self.l_gradient_b()

    def update_parameters(self):
        self.update_weights()
        self.update_bias()

    def train(self):
        self.training_features = self.load_features()
        self.N = len(self.training_features)

        res = []

        error = self.l()
        i = 0

        while error > self.threshold:
            self.update_parameters()

            if i % 5 == 0:
                print('Iteration %d: %f' % (i, error))

            error = self.l()
            i += 1

            res.append(error)

        return res

    def test(self):
        testing_features = self.load_features(False)

        for i in range(len(testing_features)):
            feature = testing_features[i]

            print('Feature %d: Calculated: %f, correct: %f, squared error: %f' %
                  (i, self.f(feature[0]), feature[1], self.squared_error(feature)))

        print('Total mean squared error: %f' % self.l(testing_features))

    @staticmethod
    def plot_results(results):
        plt.plot(results)
        plt.ylabel('Change of loss')
        plt.show()

    def run(self):
        print('Training!')

        errors = self.train()

        print('Training complete. Now testing!')

        self.test()

        print('Done!')

        self.plot_results(errors)


def main():
    linear_regression = LinearRegression(0.5)

    linear_regression.run()


if __name__ == '__main__':
    main()
