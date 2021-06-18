import operator

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import LocalOutlierFactor


class ComputeCurvature:
    def __init__(self, degree = 2, select_number = 5):
        """ Initialize some variables """
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0   # Radius of the circle
        self.x0 = np.ones(3)
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points

        self.data_number = 0
        self.select_number = select_number
        self.degree = degree
        self.polynomial_features = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()

    def func(self, p, x):
        equation = p[0]
        for i in range(1, self.degree+1):
            equation += p[i] * x**i
        return equation

    def error_func(self, p, x, y):
        return y - self.func(p, x)

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx-xc)**2 + (self.yy-yc)**2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), self.xx.size))

        ri = self.calc_r(xc, yc)
        ri[np.where(ri==0)] = 1
        df_dc[0] = (xc - self.xx)/ri                   # dR/dxc
        df_dc[1] = (yc - self.yy)/ri                   # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit_radius(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return 1.0/self.r  # Return the radius

    def linear_regression(self, x, y):
        x_poly = self.polynomial_features.fit_transform(x)
        self.model.fit(x_poly, y)
        return self.model.predict(x_poly)

    def least_sq(self, x, y, x0=0.1):
        x = np.array(x, dtype = np.float)
        y = np.array(y, dtype = np.float)
        x0 = np.ones(self.degree+1, dtype = np.float)*0.1
        parameters, success = optimize.leastsq(self.error_func, x0, args=(x, y))
        return parameters, self.func(parameters, x)

    def cur_fit(self, data_x, data_y, error_thre = 1.0):
        #selected = np.random.randint(self.data_number, size = self.select_number)
        data_number = len(data_x)
        selected = np.arange(data_number)
        x = data_x
        y = data_y
        error_max = 10.0
        para = None
        skeleton_minx = np.min(x)
        skeleton_maxx = np.max(x)
        while error_max > error_thre:
            x = x[selected]
            y = y[selected]
            #x_new = x[:, np.newaxis]
            #y_new = y[:, np.newaxis]

            para, y_pred = self.least_sq(x, y)

            difference = np.abs(y - y_pred)
            error_max = np.max(difference)
            selected = np.where(difference<error_max)
            if len(selected)<4:
                break
            #print(error_max)
            #print(x, y_pred)
            #plt.plot(x, y_pred, color='m')
            """
            plotx = np.arange(skeleton_minx-1, skeleton_maxx+2)
            ploty = self.func(para, plotx)
            plt.scatter(x, y, color='black')
            plt.plot(plotx, ploty, color='blue', linewidth=3)

            plt.xticks(())
            plt.yticks(())
            plt.show()
            """
            #selected = np.random.randint(data_number, size = 10)
        return para, x, y

    def outlier_detection(self, X):
        detector = LocalOutlierFactor(n_neighbors=3, algorithm = 'kd_tree')
        return np.where(detector.fit_predict(X)==1)

    def non_linear_fit(self, xx, yy):
        self.data_number = len(xx)
        self.xx = xx
        self.yy = yy
        data2D = np.ones((self.data_number, 2), dtype = np.float)
        data2D[:, 0] = self.xx
        data2D[:, 1] = self.yy
        outed_index = self.outlier_detection(data2D)
        outed_xx = self.xx[outed_index]
        outed_yy = self.yy[outed_index]
        cure_para, inlier_x, inlier_y = self.cur_fit(outed_xx, outed_yy, error_thre = 2.5)
        self.xx = inlier_x
        self.yy = inlier_y
        return self.fit_radius(inlier_x, inlier_y)
        #plt.show()



