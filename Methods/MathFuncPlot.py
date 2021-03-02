import numpy as np
import matplotlib.pyplot as plt
import math

def func_1():
    epsilon = 1e-10
    print(math.log(1+epsilon, 2)*1)
    print(math.log(0+epsilon, 2)*1)

    # evenly sampled time at 200ms intervals
    x0 = np.array([1, 0])
    x1 = []
    for i in range(10):
        for j in range(10):
            x1.append([i / 1.0, j / 1.0])
    # red dashes, blue squares and green triangles
    def l1_norm_sim(v1, v2):
        dis = np.abs(v1 - v2).sum()
        v1_norm = np.abs(v1).sum()
        v2_norm = np.abs(v2).sum()
        #print(v2)
        #print(dis, v1_norm, v2_norm)

        return dis / (v1_norm + v2_norm)

    results = [l1_norm_sim(x0, x) for x in x1]
    plt.plot(results)
    plt.show()


def func_2():
    x = np.arange(100)/20 - 2.5
    plt.plot(x, 1 / (1 + np.exp(x)))
    plt.show()

def func_3():
    m1 = np.array([189, 173, 171, 144, 140, 149, 150, 125, 138])
    m2 = np.array([239, 209, 98, 159, 65, 33, 91, 31, 22])
    print(m1 + m2)
    dividor = m1 + m2
    m1 = np.divide(m1, dividor)
    m2 = np.divide(m2, dividor)
    m1_norm = np.linalg.norm(m1)
    m2_norm = np.linalg.norm(m2)
    eucli_dst = np.linalg.norm(m1 - m2)
    eucli_dst = np.max(np.abs(m1 - m2))

    multiply = np.matmul(m1.transpose(), m2)
    print(m1, "\n", m2, "\n", m1_norm, m2_norm, multiply, "\n", eucli_dst)
    print(multiply/(m1_norm * m2_norm), eucli_dst, m2_norm / m1_norm )

func_3()

