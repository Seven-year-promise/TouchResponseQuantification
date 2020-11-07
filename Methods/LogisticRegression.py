import numpy as np
from Methods.TrainDataReader import generate_batch_data
from sklearn import preprocessing

class LogisticRegression:
    def __init__(self, lr=0.001, resume=False, paras_file = None, num_iter=1000000, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.epsilon = 0.0000001
        self.resume = resume
        self.theta = []
        if resume:
            print("begin initialization for the linear regression")
            self.init_theta(paras_file)

    def init_theta(self, paras_file):
        if self.resume:
            with open(paras_file, "r") as p:
                params = p.readlines()
                num = len(params)
                paras_array = np.zeros(num)
                for n in range(num):
                    paras_array[n] = float(params[n][:-1])
                self.theta = paras_array
        print("parameters initialized for linear regression")

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        epsilon = 1e-5
        m = h.shape[0]
        cost = (1 / m) * (((-y).T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
        return cost #(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def __absolute_error(self, h, y):
        #print(np.abs(h-y))
        return np.average(np.abs(h-y))

    def normlize(self, paras):
        min = np.min(paras)
        max = np.max(paras)
        internel = max - min

        paras = (paras - min) / internel

        return paras

    def fit(self, images, bboxes, well_infos, feature_size = 12*12, batch_size = 512):
        # weights initialization
        self.theta = np.zeros(feature_size + 1) +0.0001

        sample_num = len(images)
        for i in range(self.num_iter):
            batch_sample = np.random.randint(low=0, high=sample_num, size=batch_size, dtype=int)
            images_batches = [ images[i] for i in batch_sample]
            bboxes_batches = [bboxes[i] for i in batch_sample]
            well_infos_batches = [well_infos[i] for i in batch_sample]
            X, y = generate_batch_data(images = images_batches, gt_boxes = bboxes_batches,
                                       well_infos = well_infos_batches, resize=12,
                                       or_threshold = 0.05, num = batch_size, block_size = 12)
            if y.sum() == 0:
                continue
            if self.fit_intercept:
                X = self.__add_intercept(X)

            """
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.sum(np.dot(X.T, (h - y)), axis =1) / y.size
            self.theta -= self.lr * gradient
            """
            #self.theta = self.normlize(self.theta)
            #self.theta = preprocessing.normalize(self.theta.reshape(-1, 1), norm='l2')
            #self.theta = self.theta[:, 0]
            gradient = X.T @ (self.__sigmoid(X @ self.theta) - y)
            #gradient = np.average(gradient, axis = 1)

            self.theta = self.theta - (self.lr / batch_size) * gradient



            if (self.verbose == True and i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y), self.__absolute_error(h, y)} \t')
        print(self.theta)
        np.savetxt('para.txt', self.theta, delimiter='\n')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        #print(self.theta)
        prob = self.predict_prob(X)
        return prob[0] >= threshold