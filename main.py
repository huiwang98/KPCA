from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist,squareform
import numpy as np


class KPCA():
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
        self.title = f"{kernel} kernel"

    def fit_transform(self, X, y):
        if self.kernel == 'None':
            C = np.cov(X.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:, arg_max]
            K = X
        else:
            if self.kernel == 'linear':
                K = np.dot(X, X.T)
            elif self.kernel == 'log':
                dists = pdist(X) ** 0.2
                mat = squareform(dists)
                K = -np.log(1 + mat)
            elif self.kernel == 'rbf':
                dists = pdist(X) ** 2
                mat = squareform(dists)
                beta = 10
                K = np.exp(-beta * mat)
            elif self.kernel == 'sigmoid':
                K = np.tanh(np.dot(X, X.T))
            else:
                print('kernel error!')
                return None
            N = K.shape[0]
            one_n = np.ones([N, N]) / N
            K_hat = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
            eigvals, eigvecs = np.linalg.eigh(K_hat)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:,arg_max]
        X_new = np.dot(K, eigvecs_max)
        return X_new




# data
X, Y = make_circles(n_samples=400, factor=.3, noise=.05, random_state=0)

for i in range(2):  # 两类
    tmp = Y == i
    Xi = X[tmp]
    plt.scatter(Xi[:, 0], Xi[:, 1])
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure()
for i in range(2):  # 两类
    tmp = Y == i
    Xi = X_pca[tmp]
    plt.scatter(Xi[:, 0], Xi[:, 1])
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.show()

print(pca.explained_variance_ratio_)
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.figure()
plt.bar(['dim 1','dim 2'],height=pca.explained_variance_ratio_,width=0.5)
plt.title("Scree Plot (碎石图)")
plt.show()


# kpca = KPCA(kernel='linear')
# kpca = KPCA(kernel='log')
# kpca = KPCA(kernel='rbf')
kpca = KPCA(kernel='sigmoid')
X_new = kpca.fit_transform(X, Y)
for i in range(2):
    tmp = Y == i
    Xi = X_new[tmp]
    plt.scatter(Xi[:, 0], Xi[:, 1])
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.title(kpca.title)
plt.show()



print(pca.singular_values_)
pass


