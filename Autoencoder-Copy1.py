#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### KATE

from keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint


# In[ ]:





# In[ ]:


#Replaces io.utils
#Check AA lab compatibiltiy
'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

import json
#import cPickle as pickle
import pickle as pickle
import marshal as m


def dump_marshal(data, path_to_file):
    try:
        with open(path_to_file, 'w') as f:
            m.dump(data, f)
    except Exception as e:
        raise e

def load_marshal(path_to_file):
    try:
        with open(path_to_file, 'r') as f:
            data = m.load(f)
    except Exception as e:
        raise e

    return data

def dump_pickle(data, path_to_file):
    try:
        with open(path_to_file, 'w') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise e

def load_pickle(path_to_file):
    try:
        with open(path_to_file, 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        raise e

    return data

def dump_json(data, file):
    try:
        with open(file, 'w') as datafile:
            json.dump(data, datafile)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as datafile:
            data = json.load(datafile)
    except Exception as e:
        raise e

    return data

def write_file(data, file):
    try:
        with open(file, 'w') as datafile:
            for line in data:
                datafile.write(' '.join(line) + '\n')
    except Exception as e:
        raise e

def load_file(file, float_=False):
    data = []
    try:
        with open(file, 'r') as datafile:
            for line in datafile:
                content = line.strip('\n').split()
                if float_:
                    content = [float(x) for x in content]
                data.append(content)
    except Exception as e:
        raise e

    return data


# In[ ]:


#### Replaces __main__.op.utils

'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import numpy as np


def calc_ranks(x):
    """Given a list of items, return a list(in ndarray type) of ranks.
    """
    n = len(x)
    index = list(zip(*sorted(list(enumerate(x)), key=lambda d:d[1], reverse=True))[0])
    rank = np.zeros(n)
    rank[index] = range(1, n + 1)
    return rank

def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

def l1norm(x):
    return x / sum([np.abs(y) for y in x])

def vecnorm(vec, norm, epsilon=1e-3):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    """
    if norm not in ('prob', 'max1', 'logmax1'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms include 'prob',             'max1' and 'logmax1'." % norm)

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'prob':
            veclen = np.sum(np.abs(vec)) + epsilon * len(vec) # smoothing
        elif norm == 'max1':
            veclen = np.max(vec) + epsilon
        elif norm == 'logmax1':
            vec = np.log10(1. + vec)
            veclen = np.max(vec) + epsilon
        if veclen > 0.0:
            return (vec + epsilon) / veclen
        else:
            return vec
    else:
        raise ValueError('vec should be ndarray, found: %s' % type(vec))

def unitmatrix(matrix, norm='l2', axis=1):
    if norm == 'l1':
        maxtrixlen = np.sum(np.abs(matrix), axis=axis)
    if norm == 'l2':
        maxtrixlen = np.linalg.norm(matrix, axis=axis)

    if np.any(maxtrixlen <= 0):
        return matrix
    else:
        maxtrixlen = maxtrixlen.reshape(1, len(maxtrixlen)) if axis == 0 else maxtrixlen.reshape(len(maxtrixlen), 1)
        return matrix / maxtrixlen

def add_gaussian_noise(X, corruption_ratio, range_=[0, 1]):
    X_noisy = X + corruption_ratio * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = np.clip(X_noisy, range_[0], range_[1])

    return X_noisy

def add_masking_noise(X, fraction):
    assert fraction >= 0 and fraction <= 1
    X_noisy = np.copy(X)
    nrow, ncol = X.shape
    n = int(ncol * fraction)
    for i in range(nrow):
        idx_noisy  = np.random.choice(ncol, n, replace=False)
        X_noisy[i, idx_noisy] = 0

    return X_noisy

def add_salt_pepper_noise(X, fraction):
    assert fraction >= 0 and fraction <= 1
    X_noisy = np.copy(X)
    nrow, ncol = X.shape
    n = int(ncol * fraction)
    for i in range(nrow):
        idx_noisy  = np.random.choice(ncol, n, replace=False)
        X_noisy[i, idx_noisy] = np.random.binomial(1, .5, n)

    return X_noisy


# In[ ]:



### Replaced testing.visualize

'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate


class neural_net_visualizer(object):
    def __init__(self):
        pass



def heatmap(data, save_file='heatmap.png'):
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.pcolor(data, cmap=plt.cm.jet)
    plt.savefig(save_file)
    # plt.show()

def word_cloud(word_embedding_matrix, vocab, s, save_file='scatter.png'):
    words = [(i, vocab[i]) for i in s]
    model = TSNE(n_components=2, random_state=0)
    #Note that the following line might use a good chunk of RAM
    tsne_embedding = model.fit_transform(word_embedding_matrix)
    words_vectors = tsne_embedding[np.array([item[1] for item in words])]

    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        words_vectors[:, 0], words_vectors[:, 1], marker='o', cmap=plt.get_cmap('Spectral'))

    for label, x, y in zip(s, words_vectors[:, 0], words_vectors[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            fontsize=20,
            # bbox=dict(boxstyle='round,pad=1.', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '<-', connectionstyle='arc3,rad=0')
            )
    plt.show()
    # plt.savefig(save_file)

def plot_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    # markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    plt.rc('legend',**{'fontsize':30})
    classes_to_visual = list(set(classes_to_visual))
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual:
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=1, marker=markers[class_ids[c]],
                        markersize=10, label=c)
    legend = plt.legend(loc='upper right', shadow=True)
    # plt.title("tsne")
    # plt.savefig(save_file)
    plt.savefig(save_file, format='eps', dpi=2000)
    plt.show()


def plot_tsne_3d(doc_codes, doc_labels, classes_to_visual, save_file, maker_size=None, opaque=None):
    markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    plt.rc('legend',**{'fontsize':20})
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers
    while True:
        if C <= len(colors):
            break
        colors += colors

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # The problem is that the legend function don't support the type returned by a 3D scatter.
    # So you have to create a "dummy plot" with the same characteristics and put those in the legend.
    scatter_proxy = []
    for i in range(C):
        cls = classes_to_visual[i]
        idx = np.array(labels) == cls
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=colors[i], alpha=opaque[i] if opaque else 1, s=maker_size[i] if maker_size else 20, marker=markers[i], label=cls)
        scatter_proxy.append(mpl.lines.Line2D([0],[0], linestyle="none", c=colors[i], marker=markers[i], label=cls))
    ax.legend(scatter_proxy, classes_to_visual, numpoints=1)
    plt.savefig(save_file)
    plt.show()


def visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """
    # markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    plt.rc('legend',**{'fontsize':28})
    classes_to_visual = list(set(classes_to_visual))
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 1, 2

    for c in classes_to_visual:
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=1, marker=markers[class_ids[c]],
                        markersize=10, label=c)
        # plt.legend(c)
    # plt.title('Projected on the PCA components')
    # plt.xlabel('PC %s' % x_pc)
    # plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper right', shadow=True)
    # plt.savefig(save_file)
    plt.savefig(save_file, format='eps', dpi=2000)
    plt.show()

def visualize_pca_3d(doc_codes, doc_labels, classes_to_visual, save_file, maker_size=None, opaque=None):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """
    markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    plt.rc('legend',**{'fontsize':20})
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers
    while True:
        if C <= len(colors):
            break
        colors += colors

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    x_pc, y_pc, z_pc = 0, 1, 2

    # The problem is that the legend function don't support the type returned by a 3D scatter.
    # So you have to create a "dummy plot" with the same characteristics and put those in the legend.
    scatter_proxy = []
    for i in range(C):
        cls = classes_to_visual[i]
        idx = np.array(labels) == cls
        ax.scatter(X[idx, x_pc], X[idx, y_pc], X[idx, z_pc], c=colors[i], alpha=opaque[i] if opaque else 1, s=maker_size[i] if maker_size else 20, marker=markers[i], label=cls)
        scatter_proxy.append(mpl.lines.Line2D([0],[0], linestyle="none", c=colors[i], marker=markers[i], label=cls))
    ax.legend(scatter_proxy, classes_to_visual, numpoints=1)
    # plt.title('Projected on the PCA components')
    ax.set_xlabel('%sst component' % (x_pc + 1), fontsize=14)
    ax.set_ylabel('%snd component' % (y_pc + 1), fontsize=14)
    ax.set_zlabel('%srd component' % (z_pc + 1), fontsize=14)
    plt.savefig(save_file)
    plt.show()

def DBN_plot_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual.keys(), range(C)))

    codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual.keys():
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title("tsne")
    plt.savefig(save_file)
    plt.show()

def DBN_visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual.keys(), range(C)))

    codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 1, 2

    for c in classes_to_visual.keys():
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    # legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()


def reuters_visualize_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_names = classes_to_visual.keys()
    class_ids = dict(zip(class_names, range(C)))
    class_names = set(class_names)
    codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if class_names.intersection(set(doc_labels[doc]))])

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual.keys():
        idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title("tsne")
    plt.savefig(save_file)
    plt.show()

def reuters_visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_names = classes_to_visual.keys()
    class_ids = dict(zip(class_names, range(C)))
    class_names = set(class_names)
    codes, labels = zip(*[(code, class_names.intersection(set(doc_labels[doc]))) for doc, code in doc_codes.items() if len(class_names.intersection(set(doc_labels[doc]))) == 1])
    # codes = []
    # labels = []
    # for doc, code in doc_codes.items():
    #     y = set(doc_labels[doc])
    #     x = list(class_names.intersection(y))
    #     if x:
    #         codes.append(code)
    #         labels.append(x[0])
    # x = 0
    # pairs = []
    # for each in labels:
    #     if len(class_names.intersection(set(each))) > 1:
    #         x += 1
    #         pairs.append(class_names.intersection(set(each)))
    # print x


    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 0, 1

    for c in class_names:
        idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()

def get_indices(labels, c):
    idx = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        tmp = [labels[i]] if not isinstance(labels[i], (list, set)) else labels[i]
        if c in tmp:
            idx[i] = True
    return idx

def plot_info_retrieval(precisions, save_file):
    # markers = ["|", "D", "8", "v", "^", ">", "h", "H", "s", "*", "p", "d", "<"]
    markers = ["D", "p", 's', "*", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    #changing to list(zip) for Python3 compatibility SS*
    ticks = list(zip(*list(zip(*precisions))[1][0]))[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions:
        fr, pr = zip(*val)
        plt.plot(new_x, pr, linestyle='-', alpha=0.7, marker=markers[i],
                        markersize=8, label=model_name)
        i += 1
        # plt.legend(model_name)
    plt.xlabel('Fraction of Retrieved Documents')
    plt.ylabel('Precision')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

def plot_info_retrieval_by_length(precisions, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "^", "x", "D"]
    ticks = list(zip(*list(zip(*precisions))[1][0]))[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions:
        fr, pr = zip(*val)
        plt.plot(new_x, pr, linestyle='-', alpha=0.6, marker=markers[i],
                        markersize=6, label=model_name)
        i += 1
        # plt.legend(model_name)
    plt.xlabel('Document Sorted by Length')
    plt.ylabel('Precision (%)')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

def plot(x, y, x_label, y_label, save_file):
    ticks = x
    plt.xticks(range(len(ticks)), ticks, fontsize = 15)
    plt.yticks(fontsize = 15)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    plt.plot(new_x, y, linestyle='-', alpha=1.0, markersize=12, marker='p', color='b')
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=20)
    plt.savefig(save_file)
    plt.show()


# if __name__ == '__main__':
#     import sys
#     # 20news_retrieval_128D
#     precisions = [
#         ('VAE', [(0.001, 0.587348525080869), (0.002, 0.5651402500844888), (0.005, 0.5327151771489245), (0.01, 0.5014839340348453), (0.02, 0.4584269359288251), (0.05, 0.3658133556412997), (0.1, 0.2687164883998648), (0.2, 0.17739560251738207), (0.5, 0.09136516909151776), (1.0, 0.05050301672031405)]),
#         ('DocNADE', [(0.001, 0.5718148022980761), (0.002, 0.5435414956790445), (0.005, 0.5074230900538642), (0.01, 0.4746133312027964), (0.02, 0.43102761550716634), (0.05, 0.3383512940656766), (0.1, 0.25088957318799715), (0.2, 0.16893256617330504), (0.5, 0.0898931631614369), (1.0, 0.05050301672031405)]),
#         ('KATE', [(0.001, 0.5543982040264583), (0.002, 0.5213392555399969), (0.005, 0.4739445034519384), (0.01, 0.4347574243698827), (0.02, 0.3869114198299623), (0.05, 0.30403564261511706), (0.1, 0.2277761656366975), (0.2, 0.15699569840064684), (0.5, 0.08683891514289452), (1.0, 0.05050301672031405)]),
#         ('DBN', [(0.001, 0.535038381692656), (0.002, 0.5077608265340592), (0.005, 0.465912108337758), (0.01, 0.4264154357337848), (0.02, 0.37657322856108594), (0.05, 0.29198182151435376), (0.1, 0.2197600288870639), (0.2, 0.15325609847145583), (0.5, 0.08605947016611057), (1.0, 0.05050301672031403)]),
#         ('LDA',  [(0.001, 0.4867957321488915), (0.002, 0.46359774054941044), (0.005, 0.42999155982095444), (0.01, 0.40179481997753264), (0.02, 0.36320959775165296), (0.05, 0.2823678558504475), (0.1, 0.20784423242441585), (0.2, 0.14168207983103592), (0.5, 0.0800605531419018), (1.0, 0.05050301672031405)]),
#         ('Word2Vec_pre', [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)]),
#         ('Word2Vec', [(0.001, 0.3815960990682146), (0.002, 0.33990126973397794), (0.005, 0.28322016538957506), (0.01, 0.23994026666166862), (0.02, 0.1996232005978027), (0.05, 0.15086380704863955), (0.1, 0.11886672273161164), (0.2, 0.09250234660438485), (0.5, 0.06552700581695815), (1.0, 0.05050301672031405)]),
#         ('CAE',  [(0.001, 0.25605899676530824), (0.002, 0.2178643846859439), (0.005, 0.17500331917153453), (0.01, 0.14827356083073284), (0.02, 0.12567969583465557), (0.05, 0.1013255537435644), (0.1, 0.08658477146491544), (0.2, 0.07424743141317958), (0.5, 0.059710587487142405), (1.0, 0.05050301672031405)]),
#         ('KSAE', [(0.001, 0.23964418481146235), (0.002, 0.20264447448462075), (0.005, 0.16342178135194532), (0.01, 0.1395696943777457), (0.02, 0.12070563824438621), (0.05, 0.09967618984957147), (0.1, 0.08657995851945419), (0.2, 0.07516400405132624), (0.5, 0.061121338068411066), (1.0, 0.05050301672031405)]),
#         ('AE', [(0.001, 0.22827451359049142), (0.002, 0.18935571863080838), (0.005, 0.14794495865260598), (0.01, 0.12336861250406352), (0.02, 0.10404868431566056), (0.05, 0.08489066120247557), (0.1, 0.07413015988839661), (0.2, 0.06568426232571814), (0.5, 0.0563362391994616), (1.0, 0.05050301672031405)]),
#         ('DAE', [(0.001, 0.2095785255636476), (0.002, 0.17031574373581437), (0.005, 0.1300285448751998), (0.01, 0.10855864535504914), (0.02, 0.09279581161675608), (0.05, 0.07767683840981233), (0.1, 0.06946348101328252), (0.2, 0.06284691358720304), (0.5, 0.05536974244871757), (1.0, 0.05050301672031405)]),
#         ('Doc2Vec', [(0.001, 0.16486023270409583), (0.002, 0.1494834162120367), (0.005, 0.12679472346559542), (0.01, 0.11052195000447207), (0.02, 0.0953665540302444), (0.05, 0.07877845088096704), (0.1, 0.06914265711214816), (0.2, 0.06190158066520009), (0.5, 0.05536713733618202), (1.0, 0.05050301672031404)]),
#         ('NVDM', [(0.001, 0.05129628735576586), (0.002, 0.0513143919277744), (0.005, 0.0513784045216623), (0.01, 0.05057947447821584), (0.02, 0.04999729766565648), (0.05, 0.05015274063700316), (0.1, 0.050297158296132634), (0.2, 0.05053768818029844), (0.5, 0.050492220758456205), (1.0, 0.05050301672031404)])
#         ]
# 
#     # 20news_retrieval_512D
#     # precisions = {
#     # 'LDA':  [(0.001, 0.4058682952734931), (0.002, 0.37058851928739733), (0.005, 0.3309972687959938), (0.01, 0.3016110612419513), (0.02, 0.2693117036925588), (0.05, 0.2161839279252353), (0.1, 0.1702690976502034), (0.2, 0.12488871530981528), (0.5, 0.07553462776603063), (1.0, 0.05050301672031405)],
#     # 'DBN': [(0.001, 0.5553034326268522), (0.002, 0.5285147009124683), (0.005, 0.488347337076094), (0.01, 0.45024297510562405), (0.02, 0.4033891972422051), (0.05, 0.31771321417997606), (0.1, 0.23800250085341837), (0.2, 0.1643866217959289), (0.5, 0.08997211449990615), (1.0, 0.050503016720313966)],
#     # 'DocNADE': [(0.001, 0.5771737556124188), (0.002, 0.5443682711340714), (0.005, 0.5036226386465347), (0.01, 0.47000408874935945), (0.02, 0.42806973432528334), (0.05, 0.3391154672218619), (0.1, 0.2523257091581672), (0.2, 0.16856842576301645), (0.5, 0.08886419064880063), (1.0, 0.05050301672031405)],
#     # 'NVDM':  [(0.001, 0.051827354801331486), (0.002, 0.05166441365326164), (0.005, 0.05022143615810757), (0.01, 0.050504279087694), (0.02, 0.04984220717270456), (0.05, 0.050210312107871934), (0.1, 0.05031805352276878), (0.2, 0.050525479733275175), (0.5, 0.05048740951458409), (1.0, 0.05050301672031404)],
#     # 'Word2Vec_pre': [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)],
#     # 'Word2Vec': [(0.001, 0.3842755757253872), (0.002, 0.34131946120793055), (0.005, 0.28388162885972246), (0.01, 0.24101532576054588), (0.02, 0.20010492106833672), (0.05, 0.1509728403648974), (0.1, 0.118880457234514), (0.2, 0.09250264007666884), (0.5, 0.0655253160142321), (1.0, 0.05050301672031405)],
#     # 'Doc2Vec':  [(0.001, 0.2199705498961938), (0.002, 0.19429826678896794), (0.005, 0.15887688718610055), (0.01, 0.13127352793274671), (0.02, 0.1067768670780567), (0.05, 0.08213522011101435), (0.1, 0.06901493797404573), (0.2, 0.05975776562880724), (0.5, 0.05340344575184013), (1.0, 0.050503016720314126)],
#     # 'AE':  [(0.001, 0.25062762516293363), (0.002, 0.2055713802925667), (0.005, 0.15574975343297148), (0.01, 0.12843020222861343), (0.02, 0.10797588107849927), (0.05, 0.08867698410088286), (0.1, 0.07749710871105607), (0.2, 0.06824739056183722), (0.5, 0.057956525318736705), (1.0, 0.05050301672031405)],
#     # 'DAE': [(0.001, 0.2441703278134424), (0.002, 0.19592767826968308), (0.005, 0.14740440785979805), (0.01, 0.12183063178228161), (0.02, 0.1024848551783867), (0.05, 0.08429144793424902), (0.1, 0.07442198872784735), (0.2, 0.06632051023795682), (0.5, 0.05714302612312982), (1.0, 0.05050301672031405)],
#     # 'CAE':  [(0.001, 0.2564210882054672), (0.002, 0.20924660841017487), (0.005, 0.16076881496092835), (0.01, 0.1325765230591466), (0.02, 0.11132618820467256), (0.05, 0.09089712800606133), (0.1, 0.07922084751978385), (0.2, 0.06933241629113954), (0.5, 0.058349474860945535), (1.0, 0.05050301672031405)],
#     # 'VAE':  [(0.001, 0.2864384685945949), (0.002, 0.22214913339448517), (0.005, 0.15730739321751025), (0.01, 0.12415463932062157), (0.02, 0.10121123325141194), (0.05, 0.08010235972535741), (0.1, 0.06876865603310822), (0.2, 0.06069940080002707), (0.5, 0.05329987492643488), (1.0, 0.05050301672031405)],
#     # 'KSAE': [(0.001, 0.2766257905663044), (0.002, 0.23146091826389079), (0.005, 0.18327753964039104), (0.01, 0.15379102261032626), (0.02, 0.13065375342492558), (0.05, 0.10653729926356474), (0.1, 0.09147214149778002), (0.2, 0.07821235936221227), (0.5, 0.0621660351341905), (1.0, 0.05050301672031405)],
#     # 'KATE': [(0.001, 0.5370057451841842), (0.002, 0.49623424902234925), (0.005, 0.4398091950534882), (0.01, 0.39517410082761617), (0.02, 0.3440230238886337), (0.05, 0.26452986431933806), (0.1, 0.19919513465212774), (0.2, 0.14045712651660616), (0.5, 0.08193839335997657), (1.0, 0.05050301672031405)]}
# 
#     # precisions = {
#     # 'DocNADE': [(100, 0.5620457973399164), (120, 0.6721578198088268), (150, 0.6984651711924437), (200, 0.6809496236247824), (300, 0.518887505188875), (1000, 0.3119956966110817), (1500, 0.1818181818181818), (2000, 0.13636363636363635), (4000, 0.03305785123966942)],
#     # 'KCAE': [(100, 0.517573929338634), (120, 0.6815131177547284), (150, 0.7079102715466347), (200, 0.7348002316155173), (300, 0.6832710668327107), (1000, 0.6503496503496502), (1500, 0.6969696969696969), (2000, 0.8522727272727273), (4000, 0.42975206611570244)],
# 
#     # }
# 
# 
#     # plot_info_retrieval_by_length(precisions, sys.argv[1])
#     plot_info_retrieval(precisions, sys.argv[1])
# 
#     # # Effect of number of topics
#     # x = [20, 32, 64, 128, 256, 512, 1024, 1500]
#     # y = [0.546, 0.694, 0.719, 0.744, 0.747, 0.761, 0.767, 0.713]
#     # plot(x, y, 'Number of topics', 'Classification accuracy', sys.argv[1])
# 
#     # Effect of alpha
#     # x = [0.0625, 0.3, 1, 3, 6, 9, 12]
#     # y = [0.711, 0.706, 0.739, 0.738, 0.743, 0.746, 0.743]
#     # plot(x, y, r'$\alpha$', 'Classification accuracy', sys.argv[1])
# 
#     # # # Effect of k
#     # x = [2, 4, 6, 8, 16, 32, 64, 96, 128]
#     # y = [0.729, 0.728, 0.720, 0.738, 0.737,  0.744, 0.739, 0.733, 0.714]
#     # plot(x, y, r'$k$', 'Classification accuracy', sys.argv[1])
# 
#     # # scalability
#     # x = [0.5, 1, 1.5, 2]
#     # y = [6 , 12.3, 15.8,  49.5]
#     # plot(x, y, r'training set size', 'runtime (h)', sys.argv[1])
# 

# In[ ]:



### Replace from keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint

'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import numpy as np
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from keras.engine import Layer
import tensorflow as tf
from keras import initializers
import warnings

#from ..testing.visualize import heatmap
#rom .op_utils import unitmatrix replace cell 

def contractive_loss(model, lam=1e-4):
    def loss(y_true, y_pred):
        ent_loss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

        W = K.variable(value=model.encoder.get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.encoder.output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return ent_loss + contractive
    return loss


def weighted_binary_crossentropy(feature_weights):
    def loss(y_true, y_pred):
        # try:
        #     x = K.binary_crossentropy(y_pred, y_true)
        #     # y = tf.Variable(feature_weights.astype('float32'))
        #     # z = K.dot(x, y)
        #     y_true = tf.pow(y_true + 1e-5, .75)
        #     y2 = tf.div(y_true, tf.reshape(K.sum(y_true, 1), [-1, 1]))
        #     z = K.sum(tf.mul(x, y2), 1)
        # except Exception as e:
        #     print e
        #     import pdb;pdb.set_trace()
        # return z
        return K.dot(K.binary_crossentropy(y_pred, y_true), K.variable(feature_weights.astype('float32')))
    return loss


class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    # Arguments
    '''
    def __init__(self, topk, ctype, **kwargs):
        self.topk = topk
        self.ctype = ctype
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            warnings.warn("Unknown ctype, using no competition.")
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def k_comp_sigm(self, x, topk):
    #     print 'run k_comp_sigm'
    #     dim = int(x.get_shape()[1])
    #     if topk > dim:
    #         warnings.warn('topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
    #         topk = dim

    #     values, indices = tf.nn.top_k(x, topk) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

    #     # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    #     my_range = tf.expand_dims(tf.range(0, K.shape(indices)[0]), 1)  # will be [[0], [1]]
    #     my_range_repeated = tf.tile(my_range, [1, topk])  # will be [[0, 0], [1, 1]]

    #     full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    #     full_indices = tf.reshape(full_indices, [-1, 2])

    #     to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

    #     batch_size = tf.to_float(tf.shape(x)[0])
    #     tmp = 1 * batch_size * tf.reduce_sum(x - to_reset, 1, keep_dims=True) / topk

    #     res = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tmp), [-1]), default_value=0., validate_indices=False)

    #     return res

    def k_comp_tanh(self, x, topk, factor=6.26):
        print('run k_comp_tanh')
        dim = int(x.get_shape()[1])
        # batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P, topk / 2) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)


        values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)


        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        # factor = 0.
        # factor = 2. / topk
        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True) # 6.26
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)

        res = P_reset - N_reset

        return res

    # def k_comp_tanh_strict(self, x, topk):
    #     print 'run k_comp_tanh_strict'
    #     dim = int(x.get_shape()[1])
    #     # batch_size = tf.to_float(tf.shape(x)[0])
    #     if topk > dim:
    #         warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
    #         topk = dim

    #     x_abs = tf.abs(x)
    #     P = (x + x_abs) / 2 # positive part of x
    #     N = (x - x_abs) / 2 # negative part of x

    #     values, indices = tf.nn.top_k(x_abs, topk) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
    #     # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    #     my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
    #     my_range_repeated = tf.tile(my_range, [1, topk])  # will be [[0, 0], [1, 1]]
    #     full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    #     full_indices = tf.reshape(full_indices, [-1, 2])
    #     x_topk_mask = tf.sparse_to_dense(full_indices, tf.shape(x), tf.ones([tf.shape(full_indices)[0], ], tf.float32), default_value=0., validate_indices=False)

    #     P_select = tf.multiply(P, x_topk_mask)
    #     N_select = tf.multiply(-N, x_topk_mask)

    #     zero = tf.constant(0., dtype=tf.float32)
    #     P_indices = tf.cast(tf.where(tf.not_equal(P_select, zero)), tf.int32)
    #     N_indices = tf.cast(tf.where(tf.not_equal(N_select, zero)), tf.int32)
    #     P_mask = tf.sparse_to_dense(P_indices, tf.shape(x), tf.ones([tf.shape(P_indices)[0], ], tf.float32), default_value=0., validate_indices=False)
    #     N_mask = tf.sparse_to_dense(N_indices, tf.shape(x), tf.ones([tf.shape(N_indices)[0], ], tf.float32), default_value=0., validate_indices=False)

    #     alpha = 10.
    #     P_complement = alpha * tf.reduce_sum(P - P_select, 1, keep_dims=True)# / tf.cast(tf.shape(P_indices)[0], tf.float32) # 6.26
    #     N_complement = alpha * tf.reduce_sum(-N - N_select, 1, keep_dims=True)# / tf.cast(tf.shape(N_indices)[0], tf.float32)

    #     P_reset = tf.multiply(tf.add(P_select, P_complement), P_mask)
    #     N_reset = tf.multiply(tf.add(N_select, N_complement), N_mask)
    #     res = P_reset - N_reset

    #     return res

    def kSparse(self, x, topk):
        print('run regular k-sparse')
        dim = int(x.get_shape()[1])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        return res


class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)


    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, custom_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.custom_model = custom_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('CustomModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        model = self.custom_model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            model.save_weights(filepath, overwrite=True)
                        else:
                            model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    model.save_weights(filepath, overwrite=True)
                else:
                    model.save(filepath, overwrite=True)

class VisualWeights(Callback):
    def __init__(self, save_path, per_epoch=15):
        super(VisualWeights, self).__init__()
        self.per_epoch = per_epoch
        self.filename, self.ext = os.path.splitext(save_path)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        if epoch % self.per_epoch == 0:
            weights = self.model.get_weights()[0]
            # weights /= np.max(np.abs(weights))
            weights = unitmatrix(weights, axis=0) # normalize
            # weights[np.abs(weights) < 1e-2] = 0
            heatmap(weights.T, '%s_%s%s'%(self.filename, epoch, self.ext))


# In[ ]:



### Autoencoder.core.ae 
'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta
from keras.models import load_model as load_keras_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#from keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint


class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, save_model='best_model'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.save_model = save_model

        self.build()

    def build(self):
        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        if self.ctype == None:
            act = 'sigmoid'
        elif self.ctype == 'kcomp':
            act = 'tanh'
        elif self.ctype == 'ksparse':
            act = 'linear'
        else:
            raise Exception('unknown ctype: %s' % self.ctype)
        encoded_layer = Dense(self.dim, activation=act, kernel_initializer="glorot_normal", name="Encoded_Layer")
        encoded = encoded_layer(input_layer)

        if self.comp_topk:
            print('add k-competitive layer')
            encoded = KCompetitive(self.comp_topk, self.ctype)(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        decoded = Dense_tied(self.input_size, activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(outputs=decoded, inputs=input_layer)

        # this model maps an input to its encoded representation
        self.encoder = Model(outputs=encoded, inputs=input_layer)

        # create a placeholder for an encoded input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)
#removing x_val from this call 
    def fit(self, train_X, nb_epoch=50, batch_size=100, contractive=None):
        optimizer = Adadelta(lr=2.)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if contractive:
            print('Using contractive loss, lambda: %s' % contractive)
            self.autoencoder.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print('Using binary crossentropy')
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse

        self.autoencoder.fit(train_X[0], train_X[1],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        #shuffle=True,
                        shuffle=True
                      #  validation_data=(val_X[0], val_X[1]),
                        #callbacks=[
                        #            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                        #            EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                       #            CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        #]
                        )

        return self

def save_ae_model(model, model_file):
    model.save(model_file)

def load_ae_model(model_file):
    return load_keras_model(model_file, custom_objects={"KCompetitive": KCompetitive})


# In[ ]:


#### Replaces autoencoder.preprocessing 

'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import re
import string
import codecs
import numpy as np
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer as EnglishStemmer
# from nltk.tokenize import RegexpTokenizer

#from ..utils.io_utils import dump_json, load_json, write_file


def load_stopwords(file):
    stop_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                stop_words.append(line.strip('\n '))
    except Exception as e:
        raise e

    return stop_words

def init_stopwords():
    try:
        stopword_path = 'patterns/english_stopwords.txt'
        cached_stop_words = load_stopwords(os.path.join(os.path.split(__file__)[0], stopword_path))
        print('Loaded %s' % stopword_path)
    except:
        from nltk.corpus import stopwords
        cached_stop_words = stopwords.words("english")
        print('Loaded nltk.corpus.stopwords')

    return cached_stop_words

def tiny_tokenize(text, stem=False, stop_words=[]):
    words = []
    for token in wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ',             text.decode(encoding='UTF-8', errors='ignore'))):
        if not token.isdigit() and not token in stop_words:
            if stem:
                try:
                    w = EnglishStemmer().stem(token)
                except Exception as e:
                    w = token
            else:
                w = token
            words.append(w)

    return words

    # return [EnglishStemmer().stem(token) if stem else token for token in wordpunct_tokenize(
    #                     re.sub('[%s]' % re.escape(string.punctuation), ' ', text.decode(encoding='UTF-8', errors='ignore'))) if
    #                     not token.isdigit() and not token in stop_words]

def tiny_tokenize_xml(text, stem=False, stop_words=[]):
    return [EnglishStemmer().stem(token) if stem else token for token in wordpunct_tokenize(
                        re.sub('[%s]' % re.escape(string.punctuation), ' ', text.encode(encoding='ascii', errors='ignore'))) if
                        not token.isdigit() and not token in stop_words]

def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]

def count_words(docs):
    # count the number of times a word appears in a corpus
    word_freq = defaultdict(lambda: 0)
    for each in docs:
        for word, val in each.iteritems():
            word_freq[word] += val

    return word_freq

def load_data(corpus_path, recursive=False, stem=False, stop_words=False):
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc
    files = get_all_files(corpus_path, recursive)

    # word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # match only alphabet characters
    # cached_stop_words = init_stopwords()
    cached_stop_words = init_stopwords() if stop_words else []

    for filename in files:
        try:
            # with open(filename, 'r') as fp:
            with codecs.open(filename, 'r', encoding='UTF-8', errors='ignore') as fp:
                text = fp.read().lower()
                # words = [word for word in word_tokenizer.tokenize(text) if word not in cached_stop_words]
                # remove punctuations, stopwords and *unnecessary digits*, stemming
                words = tiny_tokenize(text.decode('utf-8'), stem, cached_stop_words)

                # doc_name = os.path.basename(filename)
                parent_name, child_name = os.path.split(filename)
                doc_name = os.path.split(parent_name)[-1] + '_' + child_name
                for i in range(len(words)):
                    # doc-word frequency
                    try:
                        doc_word_freq[doc_name][words[i]] += 1
                    except:
                        doc_word_freq[doc_name][words[i]] = 1
                    # word frequency
                    word_freq[words[i]] += 1
        except Exception as e:
            raise e

    return word_freq, doc_word_freq

def construct_corpus(corpus_path, training_phase, vocab_dict=None, threshold=5, topn=None, recursive=False):
    if not (training_phase or isinstance(vocab_dict, dict)):
        raise ValueError('vocab_dict must be provided if training_phase is set False')

    word_freq, doc_word_freq = load_data(corpus_path, recursive)
    if training_phase:
        vocab_dict = build_vocab(word_freq, threshold=threshold, topn=topn)

    docs = generate_bow(doc_word_freq, vocab_dict)
    new_word_freq = dict([(vocab_dict[word], freq) for word, freq in word_freq.iteritems() if word in vocab_dict])

    return docs, vocab_dict, new_word_freq

def load_corpus(corpus_path):
    corpus = load_json(corpus_path)

    return corpus

def generate_bow(doc_word_freq, vocab_dict):
    docs = {}
    for key, val in doc_word_freq.iteritems():
        word_count = {}
        for word, freq in val.iteritems():
            try:
                word_count[vocab_dict[word]] = freq
            except: # word is not in vocab, i.e., this word should be discarded
                continue
        docs[key] = word_count

    return docs

def build_vocab(word_freq, threshold=5, topn=None, start_idx=0):
    """
    threshold only take effects when topn is None.
    words are indexed by overall frequency in the dataset.
    """
    word_freq = sorted(word_freq.iteritems(), key=lambda d:d[1], reverse=True)
    if topn:
        word_freq = zip(*word_freq[:topn])[0]
        vocab_dict = dict(zip(word_freq, range(start_idx, len(word_freq) + start_idx)))
    else:
        idx = start_idx
        vocab_dict = {}
        for word, freq in word_freq:
            if freq < threshold:
                return vocab_dict
            vocab_dict[word] = idx
            idx += 1
    return vocab_dict

def construct_train_test_corpus(train_path, test_path, output, threshold=5, topn=None):
    train_docs, vocab_dict, train_word_freq = construct_corpus(train_path, True, threshold=threshold, topn=topn, recursive=True)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    dump_json(train_corpus, os.path.join(output, 'train.corpus'))
    print('Generated training corpus')

    test_docs, _, _ = construct_corpus(test_path, False, vocab_dict=vocab_dict, recursive=True)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}
    dump_json(test_corpus, os.path.join(output, 'test.corpus'))
    print('Generated test corpus')

    return train_corpus, test_corpus

def corpus2libsvm(docs, doc_labels, output):
    '''Convert the corpus format to libsvm format.
    '''
    data = []
    names = []
    for key, val in docs.iteritems():
        # label = doc_labels[key]
        label = 0
        line = label if isinstance(label, list) else [str(label)] + ["%s:%s" % (int(x) + 1, y) for x, y in val.iteritems()]
        data.append(line)
        names.append(key)
    write_file(data, output)
    write_file(names, output + '.fnames')
    return data, names

def doc2vec(doc, dim):
    vec = np.zeros(dim)
    for idx, val in doc.items():
        vec[int(idx)] = val

    return vec

def idf(docs, dim):
    vec = np.zeros((dim, 1))
    for each_doc in docs:
        for idx in each_doc.keys():
            vec[int(idx)] += 1

    return np.log10(1. + len(docs) / vec)

def vocab_weights(vocab_dict, word_freq, max_=100., ratio=.75):
    weights = np.zeros((len(vocab_dict), 1))

    for word, idx in vocab_dict.items():
        weights[idx] = word_freq[str(idx)]
    weights = np.clip(weights / max_, 0., 1.)

    return np.power(weights, ratio)

def vocab_weights_tfidf(vocab_dict, word_freq, docs, max_=100., ratio=.75):
    dim = len(vocab_dict)
    tf_vec = np.zeros((dim, 1))
    for word, idx in vocab_dict.items():
        tf_vec[idx] = 1. + np.log10(word_freq[idx]) # log normalization

    idf_vec = idf(docs, dim)
    tfidf_vec = tf_vec * idf_vec

    tfidf_vec = np.clip(tfidf_vec, 0., 4.)
    return np.power(tfidf_vec, ratio)

# # Init weights with topic modeling results
# def init_weights(topic_vocab_dist, vocab_dict, epsilon=1e-5):
#     weights = np.zeros((len(vocab_dict), len(topic_vocab_dist)))
#     for i in range(len(topic_vocab_dist)):
#         for k, v in topic_vocab_dist[i]:
#             weights[vocab_dict[k]][i] = 1. + epsilon

#     return weights

# def init_weights2(topic_vocab, vocab_dict, epsilon=1e-5):
#     weights = np.zeros((len(vocab_dict), len(topic_vocab)))
#     for i in range(len(topic_vocab)):
#         for vocab in topic_vocab[i]:
#             weights[vocab_dict[vocab]][i] = 1. / len(topic_vocab[i]) + epsilon

#     return weights

def generate_20news_doc_labels(doc_names, output):
    doc_labels = {}
    for each in doc_names:
        label = each.split('_')[0]
        doc_labels[each] = label

    dump_json(doc_labels, output)

    return doc_labels

def generate_8k_doc_labels(doc_names, output):
    doc_labels = {}
    for each in doc_names:
        label = each.split('_')[-1].replace('.txt', '')
        doc_labels[each] = label

    dump_json(doc_labels, output)

    return doc_labels

def get_8k_doc_bnames(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[-1].replace('.txt', '')

    return doc_labels

def get_8k_doc_years(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[0]

    return doc_labels

def get_8k_doc_fails(doc_names, bank_fyear):
    doc_labels = {}
    for doc in doc_names:
        fyear = bank_fyear[doc.split('-')[-1].replace('.txt', '')]
        doc_labels[doc] = 1 if fyear != 'NA' and abs(int(doc.split('-')[0]) - int(fyear)) <= 1 else 0

    return doc_labels


# In[ ]:


########## Prepare data NLPCF
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import seaborn as sns

import string
import gensim

os.chdir("/home/spenser/Downloads/case_study")

CFPB_1 = pd.read_csv("cfpb_1.csv")

CFPB_2 = pd.read_csv("cfpb_2.csv", header = None)
CFPB_2.columns = ['complaint_id', 'text']

CFPB_3 = pd.read_csv("cfpb_3.csv", header = None)
CFPB_3.columns = ['complaint_id', 'text']

CFPB_4 = pd.read_csv("cfpb_4.csv", header = None)
CFPB_4.columns = ['complaint_id', 'text']

CFPB_5 = pd.read_csv("cfpb_5.csv", header = None)
CFPB_5.columns = ['complaint_id', 'text']

CFPB_text = pd.concat([CFPB_1, CFPB_2, CFPB_3, CFPB_4, CFPB_5])

file_no_text = pd.read_csv("cfpb_triage_case_study_notext.csv")

CFPB_prod = pd.read_csv("Consumer_Complaints.csv")

#names = full_CFPB_data.columns.tolist()
#names[names.index('Complaint ID')] = 'complaint_id'
#full_CFPB_data.columns = names

CFPB_Case_Study_Joined = file_no_text.merge(CFPB_text, on = 'complaint_id', how ='left')

import string 
#text already appears to be cleansed ... but, just making sure. limited pre-processing - these assumptions can be played around with when we build classifiers

CFPB_Case_Study_Joined["text_lower"] = CFPB_Case_Study_Joined["text"].str.lower()

CFPB_Case_Study_Joined["text_lower"] = CFPB_Case_Study_Joined["text_lower"].str.replace(r'\nRevision: (\d+)\n', '') #remove digits

def remove_punctuations(text):

    for punctuation in string.punctuation:

        text = text.replace(punctuation, '')

    return text


CFPB_Case_Study_Joined["text_clean"] = CFPB_Case_Study_Joined['text_lower'].apply(remove_punctuations)  #remove punctuation


#Adding this due to finding below that pre-cleansed text is corrupt. (contains cases like can t) . Remove single stand-alone characters. ("a", "e", etc)

CFPB_Case_Study_Joined["text_clean"] = CFPB_Case_Study_Joined["text_clean"].str.replace(r'\b(?<=)[a-z](?=)\b', '') #remove single stand-alone characters.

 
cfpb_pd = CFPB_Case_Study_Joined

cfpb_pd["text_clean"] = cfpb_pd["text_clean"].str.replace(r'xx', '')
cfpb_pd["text_clean"] = cfpb_pd["text_clean"].str.replace(r'(.)\1{2,}', '')

cfpb_pd["word_count"] = cfpb_pd["text_clean"].str.count(' ') + 1

cfpb_pd = cfpb_pd[cfpb_pd["word_count"] >= 10]


# In[ ]:


cfpb_pd_sample = cfpb_pd.sample(80000, random_state=42)

fullcorpus = ' '.join(cfpb_pd_sample["text_clean"]) ###Determine vocabulary size



from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
words = set(text_to_word_sequence(fullcorpus))

print(len(words))


# In[ ]:


import gc
fullcorpus = []
CFPB_Case_Study_Joined = []
cfpb_pd = []
CFPD_text = []
gc.collect()


# In[ ]:


#### Prepare bag of words vectorization for model

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#TFIDF (Term Frequency Inverse Document Frequency will provide relative importance weightings for terms, and
# de-emphasize more common terms ... this should help with the problems noted above.)
import gc
gc.collect()
#without going through and manually picking out stop-words  like time, ask, etc, as stop words that seem to be driving the
#above categories 

#tfidf_vectorizer = TfidfVectorizer()
#tfidf = tfidf_vectorizer.fit_transform(bank_service_train["sparse"])

#tfidf_array = tfidf.toarray()
#df = tfidf_array
#Use silhouette score                                               
def cv(data):
    cv_vectorizer = CountVectorizer(min_df=40)
    train = cv_vectorizer.fit_transform(data)
    return train, cv_vectorizer

cv_array_sparse, cv_vectorizer = cv(cfpb_pd_sample["text_clean"])
#tfidf_array = tfidf_array_sparse.toarray()


# In[ ]:


cv_array = cv_array_sparse.toarray()


# In[ ]:


cv_array.shape


# In[ ]:


cv_array_sparse = []
gc.collect()


# In[ ]:


#### Sequencing and Padding Text Inputs for Input to embedding layer
from sklearn.pipeline import make_pipeline, TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
#feature_size = len(word_set)
maxlen = 8000
maxlen = 8000
max_review_length = 8000
#fullcorpus = ' '.join(Train_TfIDF_Clusters_Products['text_clean_banks']) ###Determine vocabulary size



#class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin, lower = False):
#    def __init__(self,  **kwargs):

 #       super().__init__(**kwargs)

 #   def fit(self, texts, y=None):
   #     self.fit_on_texts(texts)
   #     return self

   # def transform(self, texts, y = None):

     #   return np.array(self.texts_to_sequences(texts))
    
    

class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):

    def __init__(self,  **kwargs):

        super().__init__(**kwargs)



    def fit(self, texts, y=None):

        self.fit_on_texts(texts)

        return self



    def transform(self, texts, y = None):

        return np.array(self.texts_to_sequences(texts))
        
    
# = feature size without banks
sequencer = TextsToSequences(num_words=vocab_size)

### Function to pad sequences. Padding constrains all input vectors to be the same length.

class Padder(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen=maxlen):
        self.maxlen = maxlen
        self.max_index = None
    def fit(self, X, y=None):
        self.max_indent = pad_sequences(X, maxlen=self.maxlen).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
                    #X[X > self.max_index] = 0
        return X
padder = Padder(maxlen)

#feature_size = 80826


# In[ ]:


test_array = np.array(cfpb_pd["text_clean"][0:2])
test_array


# In[ ]:


word_list_full = []
for doc in cfpb_pd["text_clean"]:
    words = doc.split()
    for word in words:
        word_list_full.append(word)


# In[ ]:


vocab_lf = set([word for word in word_list_full])


# In[ ]:


from pprint import pprint  # pretty-printer
from collections import defaultdict

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split()]
    for document in cfpb_pd["text_clean"]
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]


# In[ ]:





# In[ ]:


from gensim import corpora
dictionary = corpora.Dictionary(texts)

print(dictionary)


# In[ ]:


corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)


# In[ ]:


texts_array


# In[ ]:


texts_array = np.array(cfpb_pd["text_clean"][0:6])
pipeline1 = make_pipeline(sequencer)
pipeline1.fit_transform(texts_array)


# In[ ]:


pipeline1.named_steps


# In[ ]:


X_docs = sequencer.fit_transform(texts)


# In[ ]:


'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np

#from autoencoder.core.ae import AutoEncoder, load_ae_model, save_ae_model replaced cell 
#from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
#from autoencoder.utils.op_utils import vecnorm, add_gaussian_noise, add_masking_noise, add_salt_pepper_noise replace cell
#from autoencoder.utils.io_utils import dump_json


#def train(args):
#corpus = load_corpus(args.input)


n_vocab, docs = len(corpus['vocab']), corpus['docs']
corpus.clear() # save memory
doc_keys = docs.keys()
#X_docs = []
#for k in doc_keys:
#    X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
#    del docs[k]

###X_docs could be replaced by doc2vec ? 
#n_vocab
#X_docs = pd.read_csv("/home/spenser/u_pd_neighbor20_mindist0.05_30components.csv")
#X_docs = np.r_[X_docs]

if args.noise == 'gs':
    X_docs_noisy = add_gaussian_noise(X_docs, 0.1)
elif args.noise == 'sp':
    X_docs_noisy = add_salt_pepper_noise(X_docs, 0.1)
    pass
elif args.noise == 'mn':
    X_docs_noisy = add_masking_noise(X_docs, 0.01)
else:
    pass

n_samples = X_docs.shape[0]
np.random.seed(0)
val_idx = np.random.choice(range(n_samples), args.n_val, replace=False)
train_idx = list(set(range(n_samples)) - set(val_idx))
X_train = X_docs[train_idx]
X_val = X_docs[val_idx]
del X_docs

if args.noise:
    # X_train_noisy = X_docs_noisy[:-n_val]
    # X_val_noisy = X_docs_noisy[-n_val:]
    X_train_noisy = X_docs_noisy[train_idx]
    X_val_noisy = X_docs_noisy[val_idx]
    print('added %s noise' % args.noise)
else:
    X_train_noisy = X_train
    X_val_noisy = X_val

start = timeit.default_timer()

ae = AutoEncoder(n_vocab, args.n_dim, comp_topk=args.comp_topk, ctype=args.ctype, save_model=args.save_model)
ae.fit([X_train_noisy, X_train], [X_val_noisy, X_val], nb_epoch=args.n_epoch,         batch_size=args.batch_size, contractive=args.contractive)

print('runtime: %ss' % (timeit.default_timer() - start))

 #   if args.output:
 #       train_doc_codes = ae.encoder.predict(X_train)
  #      val_doc_codes = ae.encoder.predict(X_val)
  #      doc_keys = np.array(doc_keys)
  #      dump_json(dict(zip(doc_keys[train_idx].tolist(), train_doc_codes.tolist())), args.output + '.train')
  #      dump_json(dict(zip(doc_keys[val_idx].tolist(), val_doc_codes.tolist())), args.output + '.val')
  #      print('Saved doc codes file to %s and %s' % (args.output + '.train', args.output + '.val'))


# In[ ]:


#else:
n_val = 20
#    pass
n_samples = cv_array.shape[0]
np.random.seed(0)
val_idx = np.random.choice(range(n_samples), n_val, replace=False)
train_idx = list(set(range(n_samples)) - set(val_idx))


# In[ ]:




X_train = cv_array[train_idx]
X_val = cv_array[val_idx]


# In[ ]:


X_docs_noisy = add_gaussian_noise(cv_array, 0.1)


# In[ ]:



X_train_noisy = X_docs_noisy[train_idx]
X_val_noisy = X_docs_noisy[val_idx]

cv_array = []

gc.collect()


# In[ ]:


X_train.shape


# In[ ]:


X_docs_noisy = []
gc.collect()


# In[ ]:


from sklearn.preprocessing import normalize
X_train = normalize(X_train, norm='l1', axis=1)
X_train_noisy = normalize(X_train_noisy, norm='l1', axis=1)


# In[ ]:


ae = AutoEncoder(6255, dim = 300, comp_topk=None, ctype='kcomp')


# In[ ]:


ae.fit([X_train_noisy, X_train], [X_val_noisy, X_val], nb_epoch=20,  batch_size=100, contractive=True)


# In[ ]:


x_test_encoded.shape


# In[ ]:


x_test_encoded = ae.encoder.predict(X_train, batch_size=100)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
#plt.colorbar()
#plt.show()
print(x_test_encoded.shape)
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(x_test_encoded)  
print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  

X_pca = pca.transform(x_test_encoded)


# In[ ]:


import pandas as pd 
import numpy as np

#from hdbscan import HDBSCAN

from hdbscan.hdbscan_ import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

from scipy.spatial.distance import cdist
from sklearn import metrics


# In[ ]:


product = pd.read_csv("/home/spenser/product.csv")
product = product.astype(str)
product.columns = ["products"]
productc = pd.Categorical(product["products"]).codes


# In[ ]:


import random
def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
    


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter3d(
    x=X_pca[:,0],
    y=X_pca[:,1],
    z=X_pca[:,2],
    marker=dict(
        size=2,
        cmax=5,
        cmin=0,
        color=productc,
        colorbar=dict(
            title="Colorbar"
        ),
        colorscale="mygbm"
    ),
    mode="markers"))




fig.show()




# In[ ]:


import gc
gc.collect()


# In[ ]:


X_train = pd.read_csv("/home/spenser/Downloads/case_study/doc2vec_300d_model3_pd.csv")
X_docs = []
#else:
X_train = np.array(X_train)
#X_val = X_docs[val_idx]

X_train_noisy = add_gaussian_noise(X_train, 0.1)

X_docs_noisy = []

cv_array = []

gc.collect()


# In[ ]:


X_train_noisy.shape


# In[ ]:


ae = AutoEncoder(300, dim = 3, comp_topk=None, ctype='kcomp')
ae.fit([X_train_noisy, X_train], nb_epoch=30 ,batch_size=100, contractive=True)


# In[ ]:


x_test_encoded = ae.encoder.predict(X_train, batch_size=100)


# In[ ]:


#x_test_encoded = ae.encoder.predict(X_train, batch_size=100)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
#plt.colorbar()
#plt.show()

import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(x_test_encoded)
X_pca = pca.transform(x_test_encoded)  

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  


# In[ ]:





# In[ ]:




import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter3d(
    x=X_pca[:,0],
    y=X_pca[:,1],
    z=X_pca[:,2],
    marker=dict(
        size=2,
        cmax=5,
        cmin=0,
        color=productc,
        colorbar=dict(
            title="Colorbar"
        ),
        colorscale="mygbm"
    ),
    mode="markers"))




fig.show()


# In[ ]:



#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
#plt.colorbar()
#plt.show()

import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X_train)
X_pca = pca.transform(X_train)  

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  


import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter3d(
    x=X_pca[:,0],
    y=X_pca[:,1],
    z=X_pca[:,2],
    marker=dict(
        size=2,
        cmax=5,
        cmin=0,
        color=productc,
        colorbar=dict(
            title="Colorbar"
        ),
        colorscale="mygbm"
    ),
    mode="markers"))




fig.show()


# In[ ]:


x_test_encoded[:,0]


# In[ ]:


x_test_encoded[:,1]


# In[ ]:




