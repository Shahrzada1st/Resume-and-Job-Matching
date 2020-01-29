import os
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(r"C:\Users\khans\Insight\My code")

def plot_mds(mean_vec):
    from sklearn.manifold import MDS
    data = mean_vec
    mds = MDS(n_components=2, random_state=1)
    pos = mds.fit_transform(data)
    xs,ys = pos[:,0], pos[:,1]
    for x, y in zip(xs, ys):
        plt.scatter(x, y)
    #    plt.text(x, y, name)
    #pos2 = mds.fit_transform(model.infer_vector(resume))
    #xs2,ys2 = pos2[:,0], pos2[:,1]
    plt.scatter(xs[-1], ys[-1], c='Red', marker='+')
    plt.text(xs[-1], ys[-1],'resume')
    plt.suptitle('MDS')
    plt.grid()
    plt.savefig('distance_MDS_improved.png')
    plt.show()

def plot_pca(mean_vec):
    from sklearn.decomposition import PCA
    #data = mean_vec
    pca = PCA(n_components=2) #, whiten=True
    X = pca.fit_transform(mean_vec)
    xs,ys =X[:,0], X[:,1]
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(xs[-1], ys[-1], c='Red', marker='+')
    plt.text(xs[-1], ys[-1],'resume')
    plt.grid()
    plt.suptitle('PCA')
    plt.savefig('distance_PCA_improved.png')
    plt.show()
