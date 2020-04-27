#!/usr/bin/env python3
# coding: utf-8

from os import path
import argparse
import time
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn

from nnet import IvectorData

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute clustering on p-vector from given data.")
    parser.add_argument("lst", help="File where data segments to be used are stored.")
    parser.add_argument("features", help="File containing features vectors.")
    parser.add_argument("output-dir", help="Gives output directory.")
    parser.add_argument("--field-separator", default=",", help="Define field separator.")
    
    parser.add_argument("--input-dim", default="400", help="Define input dim")
    options = vars(parser.parse_args())

    lst_path = options["lst"]
    features_vector_path = options["features"]
    out_dir = options["output-dir"]
    sep = options["field_separator"]
    input_dim = options["input_dim"]
    

    start = time.time()

    # extract data
    segments = IvectorData.read_lst_file(lst_path)
    features = IvectorData.read_ivector_file(features_vector_path)

    data = []
    for s in segments:
        data += [features[s]]
    data = np.stack(data)

    segment_labels = [l.split(sep)[1] for l in segments]

    # encode labels
    le = LabelEncoder()
    labels = le.fit_transform(segment_labels)
    num_classes = len(le.classes_)

    # compute KMeans clustering on data
    estimator = KMeans(
        n_clusters=num_classes,
        n_init=100,
        tol=10-6,
        algorithm="elkan",
        n_jobs=-1
    )
    estimator.fit(data)
    print("KMeans: processed {0} iterations - inertia={1:.4f}".format(estimator.n_iter_, estimator.inertia_), file=sys.stderr)

    # contains distance to each cluster for each sample
    dist_space = estimator.transform(data)
    predictions = np.argmin(dist_space, axis=1)

    # gives each cluster a name (considering most represented character)
    dataframe = pd.DataFrame({
        "label": pd.Series(list(map(lambda x: le.classes_[x], labels))),
        "prediction": pd.Series(predictions)
    })

    def find_cluster_name_fn(c):
        mask = dataframe["prediction"] == c
        return dataframe[mask]["label"].value_counts(sort=False).idxmax()

    cluster_names = list(map(find_cluster_name_fn, range(num_classes)))
    predicted_labels = le.transform(
        [cluster_names[pred] for pred in predictions])

    # F-mesure
    fscores = f1_score(labels, predicted_labels, average=None)
    fscores_str = "\n".join(map(lambda i: "{0:25s}: {1:.4f}".format(le.classes_[i], fscores[i]), range(len(fscores))))
    print("F1-scores for each classes:\n{}".format(fscores_str), file=sys.stderr)
    print("Global score: {}".format(f1_score(labels, predicted_labels, average="macro")), file=sys.stderr)
    print(f1_score(labels, predicted_labels, average="macro"), file=sys.stdout)
    with open(path.join(out_dir, "eval.log"), "w") as fd:
        print("F1-scores for each classes:\n{}".format(fscores_str), file=fd)
        print("Global score: {}".format(f1_score(labels, predicted_labels, average="macro")), file=fd)

    # evaluation metric score [0;1]
    print("{0:^20} | {1:^10}".format("CLUSTER NAME", "EVAL SCORE"), file=sys.stderr)
    for c, name in enumerate(le.classes_):

        class_mask = np.where(labels == c)
        mean_dist_class = np.mean(
            pairwise_distances(data[class_mask], n_jobs=-1))

        try:
            id_cluster = cluster_names.index(name)
        except ValueError:
            print("{0:^20} | {1:^10}".format(name, "NA"), file=sys.stderr)
            continue

        cluster_mask = np.where(predictions == id_cluster)
        mean_dist_cluster = np.mean(
            pairwise_distances(data[cluster_mask], n_jobs=-1))

        s = np.absolute(mean_dist_cluster - mean_dist_class)
        s /= np.max([mean_dist_class, mean_dist_cluster])
        print("{0:^20} | {1:^10.4f}".format(name, s), file=sys.stderr)

    # process t-SNE and plot
    tsne_estimator = TSNE()
    embeddings = tsne_estimator.fit_transform(data)
    print("t-SNE: processed {0} iterations - KL_divergence={1:.4f}".format(
        tsne_estimator.n_iter_, tsne_estimator.kl_divergence_), file=sys.stderr)

    fig, [axe1, axe2] = plt.subplots(1, 2, figsize=(10, 5))
    for c, name in enumerate(le.classes_):

        c_mask = np.where(labels == c)
        axe1.scatter(embeddings[c_mask][:, 0], embeddings[c_mask][:, 1], label=name, alpha=0.2, edgecolors=None)

        try:
            id_cluster = cluster_names.index(name)
        except ValueError:
            print("WARNING: no cluster found for {}".format(name), file=sys.stderr)
            continue
        c_mask = np.where(predictions == id_cluster)
        axe2.scatter(embeddings[c_mask][:, 0], embeddings[c_mask][:, 1], label=name, alpha=0.2, edgecolors=None)

    axe1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))
    axe1.set_title("true labels")
    axe2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35))
    axe2.set_title("predicted cluster label")

    # plt.scatter(embeddings[:,0], embeddings[:,1], c=colors)
    plt.suptitle("Kmeans Clustering")

    loc = path.join(
        out_dir,
        path.splitext(path.basename(lst_path))[0]+"_kmeans.pdf"
    )
    plt.savefig(loc, bbox_inches="tight")
    plt.close()

    print("INFO: figure saved at {}".format(loc), file=sys.stderr)

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start), file=sys.stderr)
