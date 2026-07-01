#!/usr/bin/env python3
"""
Figure factory for Chapter 8 (Clustering) and Chapter 9 (Dimensionality Reduction).
Each figure folder contains a copy of this file plus a small wrapper script.
Run the wrapper script in a figure folder to regenerate that figure.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle, FancyBboxPatch, Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib import patheffects

from sklearn.datasets import load_wine, make_blobs, make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE, Isomap

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except Exception:  # pragma: no cover
    dendrogram = None
    linkage = None

# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------
COL = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
    "ink": "#333333",
    "lightgray": "#eeeeee",
}
CLASS_COLORS = [COL["blue"], COL["orange"], COL["green"], COL["purple"], COL["red"]]


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
    })


def save_figure(fig, out_dir: str | Path, base_name: str, also_jpg: bool = False, svg: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{base_name}.png", bbox_inches="tight", facecolor="white", dpi=300)
    if also_jpg:
        fig.savefig(out_dir / f"{base_name}.jpg", bbox_inches="tight", facecolor="white", dpi=300)
    if svg:
        fig.savefig(out_dir / f"{base_name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wine_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    names = list(wine.feature_names)
    Xs = StandardScaler().fit_transform(X)
    return X, Xs, y, names


def wine_pair(pair=(9, 12)):
    X, Xs, y, names = wine_data()
    return Xs[:, pair], y, [names[i] for i in pair]


def scatter_classes(ax, X2, y, s=28, edge=True, alpha=0.9, labels=True):
    for k in sorted(np.unique(y)):
        mask = y == k
        ax.scatter(X2[mask, 0], X2[mask, 1], s=s, c=CLASS_COLORS[k % len(CLASS_COLORS)],
                   edgecolor="white" if edge else "none", linewidth=0.6, alpha=alpha,
                   label=f"class {k}" if labels else None)


def add_grid(ax):
    ax.grid(True)
    ax.set_axisbelow(True)


def plot_decision_background(ax, predict_func, xlim, ylim, alpha=0.16, cmap_colors=None):
    if cmap_colors is None:
        cmap_colors = CLASS_COLORS[:3]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300), np.linspace(ylim[0], ylim[1], 300))
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap = ListedColormap(cmap_colors)
    ax.contourf(xx, yy, Z, levels=np.arange(Z.max()+2)-0.5, cmap=cmap, alpha=alpha)


def ellipse_from_cov(ax, mean, cov, n_std=2.0, edgecolor="#333333", facecolor="none", alpha=1.0, lw=1.8, label=None):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    e = Ellipse(xy=mean, width=width, height=height, angle=angle,
                edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, lw=lw, label=label)
    ax.add_patch(e)
    return e


def make_flow_box(ax, xy, text, color, width=2.6, height=0.72, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.06",
                           facecolor=color, edgecolor="#555555", lw=1.2)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, xy1, xy2, color="#555555", lw=1.4, ms=12, style="-|>", connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=ms, lw=lw,
                        color=color, connectionstyle=connectionstyle)
    ax.add_patch(a)
    return a


def sanitize_for_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# Chapter 8: clustering figures
# -----------------------------------------------------------------------------
def fig_wine_data_feature_feature(out_dir, base):
    X, Xs, y, names = wine_data()
    pairs = [(0, 1), (9, 6)]
    titles = ["Alcohol vs. malic acid", "Color intensity vs. flavanoids"]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for ax, pair, title in zip(axs, pairs, titles):
        X2 = Xs[:, pair]
        scatter_classes(ax, X2, y)
        ax.set_xlabel(names[pair[0]].replace("_", " "))
        ax.set_ylabel(names[pair[1]].replace("_", " "))
        ax.set_title(title)
        add_grid(ax)
        sanitize_for_ax(ax)
    axs[1].legend(frameon=True, loc="best")
    fig.suptitle("Wine data shown in two feature-feature views", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, base)


def fig_clustering_vs_classification(out_dir, base):
    X, y = make_blobs(n_samples=240, centers=[(-1.5, -0.2), (1.3, 0.2), (0.0, 1.8)],
                      cluster_std=[0.55, 0.65, 0.50], random_state=7)
    km = KMeans(n_clusters=3, random_state=2, n_init=10).fit(X)
    clf = LogisticRegression(random_state=2, max_iter=2000).fit(X, y)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
    xlim = (X[:, 0].min()-0.7, X[:, 0].max()+0.7)
    ylim = (X[:, 1].min()-0.7, X[:, 1].max()+0.7)
    plot_decision_background(axs[0], km.predict, xlim, ylim)
    scatter_classes(axs[0], X, km.labels_, labels=False)
    axs[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=160,
                   c="black", edgecolor="white", linewidth=1.0, label="centroids")
    axs[0].set_title("Clustering: labels discovered")
    plot_decision_background(axs[1], clf.predict, xlim, ylim)
    scatter_classes(axs[1], X, y, labels=True)
    axs[1].set_title("Classification: labels supplied")
    for ax in axs:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
        add_grid(ax); sanitize_for_ax(ax)
    axs[0].legend(frameon=True, loc="lower right")
    axs[1].legend(frameon=True, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_dir, base)


def fig_clustering_methods(out_dir, base):
    X1, _ = make_blobs(n_samples=300, centers=4, cluster_std=[0.55, 0.6, 0.45, 0.7], random_state=2)
    X2, _ = make_moons(n_samples=300, noise=0.075, random_state=4)
    X2 = X2 * np.array([2.1, 1.5])
    methods = [
        ("K-means", X1, KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(X1)),
        ("Gaussian mixture", X1, GaussianMixture(n_components=4, random_state=0).fit_predict(X1)),
        ("DBSCAN", X2, DBSCAN(eps=0.25, min_samples=6).fit_predict(X2)),
        ("Agglomerative", X1, AgglomerativeClustering(n_clusters=4).fit_predict(X1)),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.2))
    for ax, (title, X, labels) in zip(axs.ravel(), methods):
        unique = sorted(set(labels))
        for j, lab in enumerate(unique):
            mask = labels == lab
            color = "#bbbbbb" if lab == -1 else CLASS_COLORS[j % len(CLASS_COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], s=20, c=color, edgecolor="white", linewidth=0.35)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(True)
    fig.suptitle("Different clustering algorithms impose different structure", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, base)


def fig_kmeans_wine_data(out_dir, base):
    X2, y, pair_names = wine_pair((9, 12))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    scatter_classes(ax, X2, y)
    ax.set_xlabel("standardized " + pair_names[0].replace("_", " "))
    ax.set_ylabel("standardized " + pair_names[1].replace("_", " "))
    ax.set_title("Wine measurements before clustering")
    ax.legend(frameon=True, loc="best")
    add_grid(ax); sanitize_for_ax(ax)
    save_figure(fig, out_dir, base, also_jpg=True)


def fig_kmeans_decision_regions(out_dir, base):
    X2, y, pair_names = wine_pair((9, 12))
    km = KMeans(n_clusters=3, random_state=4, n_init=20).fit(X2)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xlim=(X2[:,0].min()-0.5, X2[:,0].max()+0.5); ylim=(X2[:,1].min()-0.5, X2[:,1].max()+0.5)
    plot_decision_background(ax, km.predict, xlim, ylim, alpha=0.18)
    scatter_classes(ax, X2, km.labels_, labels=False)
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c="black", marker="X", s=180,
               edgecolor="white", linewidth=1.0, label="centroids")
    ax.set_title("K-means decision regions on Wine features")
    ax.set_xlabel("standardized color intensity")
    ax.set_ylabel("standardized proline")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.legend(frameon=True)
    add_grid(ax); sanitize_for_ax(ax)
    save_figure(fig, out_dir, base, also_jpg=True)


def kmeans_manual_iterations(X, centers, steps=4):
    history=[]
    centers=centers.copy()
    for _ in range(steps):
        d=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels=d.argmin(axis=1)
        history.append((centers.copy(), labels.copy()))
        for j in range(centers.shape[0]):
            if np.any(labels==j):
                centers[j]=X[labels==j].mean(axis=0)
    d=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    labels=d.argmin(axis=1)
    history.append((centers.copy(), labels.copy()))
    return history


def fig_kmeans_centroid_updates(out_dir, base):
    X2, y, _ = wine_pair((9, 12))
    init = np.array([[-1.8, 1.5], [0.8, -0.8], [2.2, 2.5]])
    hist = kmeans_manual_iterations(X2, init, steps=3)
    fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.4), sharex=True, sharey=True)
    for t, ax in enumerate(axs):
        centers, labels = hist[t]
        for j in range(3):
            ax.scatter(X2[labels==j,0], X2[labels==j,1], s=16, color=CLASS_COLORS[j], alpha=0.70,
                       edgecolor="white", linewidth=0.25)
        ax.scatter(centers[:,0], centers[:,1], marker="X", s=140, color="black", edgecolor="white", linewidth=0.8)
        ax.set_title("initial" if t==0 else f"iteration {t}")
        ax.set_xlabel("color intensity")
        add_grid(ax); sanitize_for_ax(ax)
    axs[0].set_ylabel("proline")
    fig.suptitle("Centroids move toward the mean of their assigned points", y=1.05)
    fig.tight_layout()
    save_figure(fig, out_dir, base, also_jpg=True)


def fig_kmeans_random_starts(out_dir, base):
    X2, _, _ = wine_pair((9, 12))
    states=[1, 4, 12]
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.5), sharex=True, sharey=True)
    best_inertia=1e99; best_idx=0
    results=[]
    for s in states:
        km=KMeans(n_clusters=3, init="random", n_init=1, random_state=s, max_iter=300).fit(X2)
        results.append(km)
        if km.inertia_ < best_inertia:
            best_inertia=km.inertia_; best_idx=len(results)-1
    for i,(ax,km,s) in enumerate(zip(axs,results,states)):
        for j in range(3):
            ax.scatter(X2[km.labels_==j,0], X2[km.labels_==j,1], s=16, color=CLASS_COLORS[j], alpha=0.75,
                       edgecolor="white", linewidth=0.25)
        ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker="X", s=150, color="black", edgecolor="white", linewidth=0.8)
        ax.set_title(f"start {s}\ninertia={km.inertia_:.1f}" + ("\nlowest" if i==best_idx else ""))
        ax.set_xlabel("color intensity")
        add_grid(ax); sanitize_for_ax(ax)
    axs[0].set_ylabel("proline")
    fig.suptitle("Different random starts can produce different K-means solutions", y=1.05)
    fig.tight_layout()
    save_figure(fig, out_dir, base, also_jpg=True)


def fig_kmeans_cluster_diagnostics(out_dir, base):
    _, Xs, _, _ = wine_data()
    ks=range(2,9)
    inertias=[]; sils=[]
    for k in ks:
        labels=KMeans(n_clusters=k, random_state=2, n_init=20).fit_predict(Xs)
        km=KMeans(n_clusters=k, random_state=2, n_init=20).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, labels))
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axs[0].plot(list(ks), inertias, marker="o", lw=2.0, color=COL["blue"])
    axs[0].set_title("Elbow plot")
    axs[0].set_xlabel("number of clusters k"); axs[0].set_ylabel("inertia")
    axs[1].plot(list(ks), sils, marker="o", lw=2.0, color=COL["orange"])
    axs[1].set_title("Silhouette score")
    axs[1].set_xlabel("number of clusters k"); axs[1].set_ylabel("mean silhouette")
    for ax in axs:
        add_grid(ax); sanitize_for_ax(ax)
    fig.suptitle("Cluster diagnostics on standardized Wine data", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_dir, base, also_jpg=True)


def fig_kmeans_image_segmentation(out_dir, base):
    rng=np.random.default_rng(0)
    h,w=90,130
    yy,xx=np.mgrid[0:h,0:w]
    img=np.zeros((h,w,3),float)
    img[...,0]=0.25+0.55*(xx/w)
    img[...,1]=0.25+0.55*(yy/h)
    img[...,2]=0.35+0.25*np.sin(xx/18)
    for cx,cy,col,rad in [(55,55,(0.95,0.35,0.25),35),(140,78,(0.2,0.55,0.95),38),(110,105,(0.2,0.8,0.45),28)]:
        mask=(xx-cx)**2+(yy-cy)**2<rad**2
        img[mask]=0.65*img[mask]+0.35*np.array(col)
    img=np.clip(img+rng.normal(0,0.025,img.shape),0,1)
    pixels=img.reshape(-1,3)
    km=MiniBatchKMeans(n_clusters=5, random_state=0, n_init=3, batch_size=1024).fit(pixels)
    centers=km.cluster_centers_
    seg=centers[km.labels_].reshape(img.shape)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.4))
    axs[0].imshow(img); axs[0].set_title("original synthetic image")
    axs[1].imshow(seg); axs[1].set_title("K-means color segments")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
    fig.suptitle("Image segmentation as clustering pixels by color", y=1.03)
    fig.tight_layout()
    save_figure(fig,out_dir,base)


def fig_hierarchical_agglomerative_clustering(out_dir, base):
    X2, y, _ = wine_pair((9, 12))
    # choose representative observations across the scatter
    idx=np.array([15,25,60,80,110,120,150,170])
    Xs=X2[idx]
    fig, ax = plt.subplots(figsize=(8.6,4.5))
    if linkage is not None:
        Z=linkage(Xs, method="ward")
        dendrogram(Z, labels=[f"wine {i+1}" for i in range(len(idx))], ax=ax,
                   color_threshold=None, above_threshold_color=COL["gray"])
        ax.set_title("Hierarchical agglomerative clustering")
        ax.set_ylabel("merge distance")
    else:
        ax.text(0.5,0.5,"Hierarchical clustering dendrogram",ha="center",va="center")
    add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout()
    save_figure(fig,out_dir,base)


def fig_dbscan_vs_k_means(out_dir, base):
    X,y=make_moons(n_samples=320, noise=0.07, random_state=3)
    X=X*np.array([2.0,1.3])
    km=KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X)
    db=DBSCAN(eps=0.22, min_samples=6).fit_predict(X)
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.8),sharex=True,sharey=True)
    for ax,labels,title in [(axs[0],km,"K-means partitions by centroid distance"),(axs[1],db,"DBSCAN follows dense neighborhoods")]:
        for j, lab in enumerate(sorted(set(labels))):
            mask=labels==lab
            color="#bbbbbb" if lab==-1 else CLASS_COLORS[j]
            ax.scatter(X[mask,0],X[mask,1],s=20,c=color,edgecolor="white",linewidth=0.3)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(True)
    fig.tight_layout()
    save_figure(fig,out_dir,base)


def fig_dbscan_eps_too_small(out_dir, base):
    X,y=make_moons(n_samples=300, noise=0.075, random_state=5)
    X=X*np.array([2.0,1.3])
    labels=DBSCAN(eps=0.08,min_samples=6).fit_predict(X)
    fig,ax=plt.subplots(figsize=(6.2,4.1))
    for lab in sorted(set(labels)):
        mask=labels==lab
        color="#bbbbbb" if lab==-1 else CLASS_COLORS[lab % len(CLASS_COLORS)]
        ax.scatter(X[mask,0],X[mask,1],s=22,c=color,edgecolor="white",linewidth=0.3,label="noise" if lab==-1 else f"cluster {lab}")
    ax.set_title(r"DBSCAN with $\epsilon$ too small")
    ax.text(0.02,0.98,"many points become noise",transform=ax.transAxes,ha="left",va="top",
            bbox=dict(facecolor="white",edgecolor="#777777",boxstyle="round,pad=0.25"))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=True,loc="lower right")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dbscan_eps_neighborhoods(out_dir, base):
    X,y=make_moons(n_samples=240, noise=0.065, random_state=6)
    X=X*np.array([2.0,1.3])
    eps=0.22
    db=DBSCAN(eps=eps,min_samples=5).fit(X)
    labels=db.labels_
    fig,ax=plt.subplots(figsize=(6.4,4.3))
    for lab in sorted(set(labels)):
        mask=labels==lab
        color="#bbbbbb" if lab==-1 else CLASS_COLORS[lab % len(CLASS_COLORS)]
        ax.scatter(X[mask,0],X[mask,1],s=22,c=color,edgecolor="white",linewidth=0.3)
    # show selected neighborhoods
    core_idx=db.core_sample_indices_[::max(1,len(db.core_sample_indices_)//8)][:8]
    for idx in core_idx:
        circ=Circle(X[idx],eps,facecolor="none",edgecolor="#555555",lw=1.0,alpha=0.55)
        ax.add_patch(circ)
    ax.set_title(r"DBSCAN $\epsilon$-neighborhoods connect core points")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_gaussian_general_idea(out_dir, base):
    X2,y,_=wine_pair((9,12))
    gm=GaussianMixture(n_components=3,covariance_type="full",random_state=2).fit(X2)
    labels=gm.predict(X2)
    fig,ax=plt.subplots(figsize=(6.4,4.7))
    scatter_classes(ax,X2,labels,labels=False)
    for j,(mean,cov) in enumerate(zip(gm.means_,gm.covariances_)):
        ellipse_from_cov(ax,mean,cov,n_std=2,edgecolor=CLASS_COLORS[j],lw=2.5)
        ax.scatter(mean[0],mean[1],marker="X",s=150,c=CLASS_COLORS[j],edgecolor="white",linewidth=1.0)
    ax.set_title("Gaussian components model elliptical clouds")
    ax.set_xlabel("standardized color intensity"); ax.set_ylabel("standardized proline")
    add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_gaussian_mixture_model(out_dir, base):
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    ax.set_xlim(-5,5); ax.set_ylim(-3,3); ax.axis("off")
    make_flow_box(ax,(-3.4,1.9),"Choose number of\ncomponents k", "#fff2cc")
    make_flow_box(ax,(0,1.9),"Set mixture parameters\n" + r"$\{\pi_j, \mu_j, \Sigma_j\}$", "#fde2bd", width=3.2)
    make_flow_box(ax,(-3.4,0.2),"Hidden component\n" + r"$z_i \sim \mathrm{Categorical}(\pi)$", "#d9edf7", width=3.1)
    make_flow_box(ax,(0,0.2),"Generate features\n" + r"$x_i \sim \mathcal{N}(\mu_{z_i},\Sigma_{z_i})$", "#dff0d8", width=3.4)
    make_flow_box(ax,(3.4,0.2),"Observed data\n" + r"$X=\{x_1,\ldots,x_n\}$", "#e6e0f8", width=2.8)
    make_flow_box(ax,(3.4,-1.6),"Fit GMM to X\nsoft memberships\n" + r"$\gamma_{ij}=P(z_i=j\mid x_i)$", "#f8d7da", width=3.2, height=1.0)
    arrow(ax,(-2.1,1.9),(-1.65,1.9)); arrow(ax,(-3.4,1.55),(-3.4,0.58)); arrow(ax,(0,1.55),(0,0.58)); arrow(ax,(-1.85,0.2),(-1.72,0.2)); arrow(ax,(1.75,0.2),(2.0,0.2)); arrow(ax,(3.4,-0.18),(3.4,-1.05))
    arrow(ax,(4.95,-1.5),(1.75,1.75),style="-|>",connectionstyle="arc3,rad=0.35",color=COL["gray"],lw=1.2)
    ax.text(4.55,1.05,"parameter updates\nduring fitting",ha="center",va="center",fontsize=9,color=COL["gray"])
    ax.text(0,-2.55,"The observed wine measurements are visible; the component identity is hidden.",ha="center",fontsize=10)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_gaussian_mixture_cluster(out_dir, base):
    X2,y,_=wine_pair((9,12))
    gm=GaussianMixture(n_components=3,covariance_type="full",random_state=0).fit(X2)
    labels=gm.predict(X2)
    fig,ax=plt.subplots(figsize=(6.3,4.6))
    xlim=(X2[:,0].min()-0.6,X2[:,0].max()+0.6); ylim=(X2[:,1].min()-0.6,X2[:,1].max()+0.6)
    xx,yy=np.meshgrid(np.linspace(*xlim,250),np.linspace(*ylim,250))
    grid=np.c_[xx.ravel(),yy.ravel()]
    Z=gm.predict(grid).reshape(xx.shape)
    ax.contourf(xx,yy,Z,levels=np.arange(4)-0.5,colors=CLASS_COLORS[:3],alpha=0.13)
    scatter_classes(ax,X2,labels,labels=False)
    for j in range(3):
        ellipse_from_cov(ax,gm.means_[j],gm.covariances_[j],edgecolor=CLASS_COLORS[j],lw=2.2)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Gaussian mixture clustering on Wine features")
    ax.set_xlabel("standardized color intensity"); ax.set_ylabel("standardized proline")
    add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_gaussian_mixture_covariance(out_dir, base):
    X2,y,_=wine_pair((9,12))
    models=[("tied covariance",GaussianMixture(n_components=3,covariance_type="tied",random_state=0).fit(X2)),
            ("spherical covariance",GaussianMixture(n_components=3,covariance_type="spherical",random_state=0).fit(X2))]
    fig,axs=plt.subplots(1,2,figsize=(9.0,4.0),sharex=True,sharey=True)
    for ax,(title,gm) in zip(axs,models):
        labels=gm.predict(X2)
        scatter_classes(ax,X2,labels,labels=False,s=22)
        for j in range(3):
            if gm.covariance_type=="tied": cov=gm.covariances_
            elif gm.covariance_type=="spherical": cov=np.eye(2)*gm.covariances_[j]
            else: cov=gm.covariances_[j]
            ellipse_from_cov(ax,gm.means_[j],cov,edgecolor=CLASS_COLORS[j],lw=2.2)
        ax.set_title(title); ax.set_xlabel("color intensity")
        add_grid(ax); sanitize_for_ax(ax)
    axs[0].set_ylabel("proline")
    fig.suptitle("Covariance assumptions control GMM flexibility",y=1.03)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_gaussian_anomaly_detection(out_dir, base):
    X2,y,_=wine_pair((9,12))
    gm=GaussianMixture(n_components=3,covariance_type="full",random_state=2).fit(X2)
    scores=gm.score_samples(X2)
    cutoff=np.percentile(scores,8)
    anomalies=scores<cutoff
    fig,ax=plt.subplots(figsize=(6.6,4.8))
    xlim=(X2[:,0].min()-0.7,X2[:,0].max()+0.7); ylim=(X2[:,1].min()-0.7,X2[:,1].max()+0.7)
    xx,yy=np.meshgrid(np.linspace(*xlim,280),np.linspace(*ylim,280))
    grid=np.c_[xx.ravel(),yy.ravel()]
    dens=np.exp(gm.score_samples(grid)).reshape(xx.shape)
    ax.contourf(xx,yy,dens,levels=12,cmap="Blues",alpha=0.35)
    ax.contour(xx,yy,dens,levels=7,colors="#555555",linewidths=0.7,alpha=0.7)
    ax.scatter(X2[~anomalies,0],X2[~anomalies,1],s=24,c="#4c78a8",edgecolor="white",linewidth=0.35,label="typical")
    ax.scatter(X2[anomalies,0],X2[anomalies,1],s=55,c=COL["red"],edgecolor="white",linewidth=0.7,label="low likelihood")
    ax.set_title("Anomalies have low probability under the fitted mixture")
    ax.set_xlabel("standardized color intensity"); ax.set_ylabel("standardized proline")
    ax.legend(frameon=True); add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


# -----------------------------------------------------------------------------
# Chapter 9: dimensionality reduction figures
# -----------------------------------------------------------------------------
def fig_dimred_overview(out_dir, base):
    fig,ax=plt.subplots(figsize=(8.8,4.4))
    ax.set_xlim(-4.8,4.8); ax.set_ylim(-2.2,2.2); ax.axis("off")
    make_flow_box(ax,(-3.4,0),"High-dimensional\nmeasurements\n" + r"$X\in\mathbb{R}^{m\times d}$","#d9edf7",width=2.6,height=1.0)
    make_flow_box(ax,(0,0),"Dimensionality\nreduction\nfit transform","#fff2cc",width=2.5,height=1.0)
    make_flow_box(ax,(3.4,0),"Low-dimensional\nscores\n" + r"$Z\in\mathbb{R}^{m\times r}$","#dff0d8",width=2.6,height=1.0)
    arrow(ax,(-2.05,0),(-1.35,0)); arrow(ax,(1.35,0),(2.05,0))
    items=[("visualize",-3.6,-1.35), ("compress",-1.8,-1.35), ("denoise",0,-1.35), ("preprocess",1.8,-1.35), ("monitor",3.6,-1.35)]
    for text,x,y in items:
        ax.text(x,y,text,ha="center",va="center",fontsize=10,bbox=dict(facecolor="#f7f7f7",edgecolor="#aaaaaa",boxstyle="round,pad=0.25"))
    ax.text(0,1.5,"Keep the important structure while using fewer coordinates",ha="center",fontsize=13,weight="bold")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_wine_feature_matrix(out_dir, base):
    X, Xs, y, names=wine_data()
    order=np.argsort(y)
    Xplot=Xs[order]
    yord=y[order]
    fig=plt.figure(figsize=(9.2,5.2))
    gs=fig.add_gridspec(1,2,width_ratios=[0.18,5.0],wspace=0.03)
    ax0=fig.add_subplot(gs[0,0]); ax=fig.add_subplot(gs[0,1])
    ax0.imshow(yord[:,None],aspect="auto",cmap=ListedColormap(CLASS_COLORS[:3]))
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_title("class",fontsize=9)
    im=ax.imshow(Xplot,aspect="auto",cmap="coolwarm",vmin=-3,vmax=3)
    ax.set_title("Standardized Wine feature matrix")
    ax.set_xlabel("measured feature"); ax.set_ylabel("wine sample, sorted by class")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_"," ") for n in names],rotation=60,ha="right",fontsize=7)
    ax.set_yticks([])
    cbar=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02)
    cbar.set_label("standardized value")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_pca_shadow_projection(out_dir, base):
    rng=np.random.default_rng(8)
    cov=np.array([[2.5,1.4],[1.4,1.1]])
    X=rng.multivariate_normal([0,0],cov,size=60)
    vals,vecs=np.linalg.eigh(np.cov(X.T)); order=vals.argsort()[::-1]; vecs=vecs[:,order]
    directions=[vecs[:,1],vecs[:,0]]
    titles=["short shadow: weak direction","long shadow: first principal component"]
    fig,axs=plt.subplots(2,1,figsize=(7.2,6.2))
    for ax, v, title in zip(axs,directions,titles):
        u=v/np.linalg.norm(v)
        # light cone background
        ax.add_patch(Polygon([(-4,2.0),(-1.0,0.3),(-4,-1.4)],closed=True,facecolor="#fff3a8",edgecolor="none",alpha=0.85))
        ax.add_patch(Rectangle((-4.6,-0.25),0.5,0.5,facecolor="#cfcfcf",edgecolor="#777777"))
        ax.add_patch(Polygon([(-4.1,-0.65),(-3.35,-0.35),(-3.35,0.35),(-4.1,0.65)],closed=True,facecolor="#e0e0e0",edgecolor="#777777"))
        # data cloud
        ax.scatter(X[:,0],X[:,1],s=24,c=[CLASS_COLORS[i%3] for i in range(len(X))],edgecolor="white",linewidth=0.35,zorder=3)
        # projection line and projected points
        t=np.linspace(-3.0,3.0,2)
        ax.plot(t*u[0],t*u[1],color="#333333",lw=2.0,zorder=2)
        proj=(X@u)[:,None]*u[None,:]
        for p,q in zip(X[::4],proj[::4]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color="#666666",lw=0.4,alpha=0.5)
        ax.scatter(proj[:,0],proj[:,1],s=12,c="#333333",alpha=0.6,zorder=4)
        # screen shadow
        screen_x=3.55
        shadow=(X@u)
        s_scaled=(shadow-shadow.mean())/shadow.std()*0.7
        ax.plot([screen_x,screen_x],[-1.8,1.8],color="#555555",lw=2)
        ax.scatter(np.full_like(s_scaled,screen_x),s_scaled,s=10,c="#888888",alpha=0.55)
        ax.text(screen_x+0.15,0,"projected\nshadow",va="center",fontsize=9)
        ax.set_title(title)
        ax.set_xlim(-4.7,4.5); ax.set_ylim(-2.3,2.3); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle("Principal Component Analysis (PCA): rotate to get the largest shadow",y=0.98,fontsize=13,weight="bold")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_pca_projection_geometry(out_dir, base):
    rng=np.random.default_rng(4)
    cov=np.array([[3.0,1.65],[1.65,1.25]])
    X=rng.multivariate_normal([0,0],cov,size=220)
    pca=PCA(n_components=2).fit(X)
    comps=pca.components_
    mean=X.mean(axis=0)
    u=comps[0]
    proj=mean+(X-mean)@u[:,None]*u[None,:]
    fig,ax=plt.subplots(figsize=(6.4,5.2))
    ax.scatter(X[:,0],X[:,1],s=18,c=COL["blue"],alpha=0.45,edgecolor="white",linewidth=0.25,label="data")
    for p,q in zip(X[::6],proj[::6]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color="#888888",lw=0.5,alpha=0.45)
    ax.scatter(proj[:,0],proj[:,1],s=10,c=COL["orange"],alpha=0.65,label="projection")
    for i,color,label in [(0,COL["red"],"PC 1"),(1,COL["green"],"PC 2")]:
        v=comps[i]*np.sqrt(pca.explained_variance_[i])*2.2
        ax.arrow(mean[0],mean[1],v[0],v[1],head_width=0.13,head_length=0.22,fc=color,ec=color,lw=2.2,length_includes_head=True)
        ax.text(mean[0]+v[0]*1.08,mean[1]+v[1]*1.08,label,color=color,weight="bold")
    ax.set_title("PCA rotates the coordinate system")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.legend(frameon=True); add_grid(ax); sanitize_for_ax(ax); ax.set_aspect("equal",adjustable="box")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_covariance_matrix_structure(out_dir, base):
    n=6
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j: M[i,j]=2
            elif i<j: M[i,j]=1
            else: M[i,j]=0.55
    colors=ListedColormap(["#e8e8e8","#c7ddf2","#ffd6a5"])
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    ax.imshow(M,cmap=colors,vmin=0,vmax=2)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"feature {i+1}" for i in range(n)],rotation=45,ha="right")
    ax.set_yticklabels([f"feature {i+1}" for i in range(n)])
    for i in range(n+1):
        ax.axhline(i-0.5,color="white",lw=1.5); ax.axvline(i-0.5,color="white",lw=1.5)
    for i in range(n):
        for j in range(n):
            text="var" if i==j else "cov"
            ax.text(j,i,text,ha="center",va="center",fontsize=8,color="#333333")
    ax.plot([-0.5,n-0.5],[-0.5,n-0.5],color="#333333",lw=2.0)
    ax.text(n-0.1,0.5,"symmetric:\n$S_{ij}=S_{ji}$",ha="left",va="center",fontsize=11,
            bbox=dict(facecolor="white",edgecolor="#777777",boxstyle="round,pad=0.3"))
    ax.text(1.1,-1.15,"diagonal = feature variances",ha="center",va="center",fontsize=10,
            bbox=dict(facecolor="#ffd6a5",edgecolor="#777777",boxstyle="round,pad=0.25"))
    ax.text(4.2,-1.15,"off-diagonal = feature covariances",ha="center",va="center",fontsize=10,
            bbox=dict(facecolor="#c7ddf2",edgecolor="#777777",boxstyle="round,pad=0.25"))
    ax.set_xlim(-0.5,n+1.7); ax.set_ylim(n-0.5,-1.55)
    ax.set_title("Covariance matrix structure")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_wine_covariance_heatmap(out_dir, base):
    X, Xs, y, names=wine_data()
    S=np.cov(Xs,rowvar=False)
    fig,ax=plt.subplots(figsize=(7.8,6.4))
    im=ax.imshow(S,cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_title("Covariance matrix for standardized Wine features")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    labels=[n.replace("_"," ") for n in names]
    ax.set_xticklabels(labels,rotation=60,ha="right",fontsize=7)
    ax.set_yticklabels(labels,fontsize=7)
    for i in range(len(names)):
        ax.add_patch(Rectangle((i-0.5,i-0.5),1,1,fill=False,edgecolor="#222222",lw=0.8))
    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cbar.set_label("covariance")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_wine_scree_plot(out_dir, base):
    _,Xs,_,_=wine_data()
    pca=PCA().fit(Xs)
    ev=pca.explained_variance_ratio_
    cum=np.cumsum(ev)
    x=np.arange(1,len(ev)+1)
    fig,ax1=plt.subplots(figsize=(7.4,4.6))
    ax1.bar(x,ev,color=COL["blue"],alpha=0.75,label="individual")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance ratio")
    ax1.set_xticks(x)
    ax2=ax1.twinx()
    ax2.plot(x,cum,marker="o",color=COL["orange"],lw=2.2,label="cumulative")
    ax2.set_ylabel("cumulative explained variance")
    ax1.set_title("Scree plot for Wine data")
    ax1.grid(True,axis="y")
    ax2.set_ylim(0,1.05)
    lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2,labels+labels2,frameon=True,loc="center right")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_wine_pca_loadings(out_dir, base):
    _,Xs,_,names=wine_data()
    pca=PCA(n_components=3).fit(Xs)
    comps=pca.components_.T
    fig,ax=plt.subplots(figsize=(7.8,5.2))
    im=ax.imshow(comps,cmap="coolwarm",vmin=-0.55,vmax=0.55,aspect="auto")
    ax.set_title("PCA loadings for Wine features")
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_"," ") for n in names],fontsize=8)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["PC 1","PC 2","PC 3"])
    for i in range(comps.shape[0]):
        for j in range(comps.shape[1]):
            ax.text(j,i,f"{comps[i,j]:.2f}",ha="center",va="center",fontsize=7)
    cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03); cbar.set_label("loading")
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_wine_pca_scores(out_dir, base):
    _,Xs,y,_=wine_data()
    Z=PCA(n_components=2).fit_transform(Xs)
    fig,ax=plt.subplots(figsize=(6.3,4.8))
    scatter_classes(ax,Z,y,s=35)
    ax.set_title("Wine samples projected onto PC 1 and PC 2")
    ax.set_xlabel("PC 1 score"); ax.set_ylabel("PC 2 score")
    ax.legend(frameon=True,loc="best")
    add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_pca_reconstruction_error(out_dir, base):
    _,Xs,_,_=wine_data()
    errs=[]
    dims=range(1,Xs.shape[1]+1)
    for r in dims:
        pca=PCA(n_components=r).fit(Xs)
        Xrec=pca.inverse_transform(pca.transform(Xs))
        errs.append(np.mean((Xs-Xrec)**2))
    fig,ax=plt.subplots(figsize=(6.7,4.4))
    ax.plot(list(dims),errs,marker="o",lw=2.2,color=COL["blue"])
    ax.set_title("PCA reconstruction error decreases as components are kept")
    ax.set_xlabel("number of retained principal components")
    ax.set_ylabel("mean squared reconstruction error")
    ax.set_xticks(list(dims))
    add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_pca_reconstruction_denoising(out_dir, base):
    rng=np.random.default_rng(2)
    m=90; n=120
    t=np.linspace(0,1,n)
    amps=rng.uniform(0.8,1.2,m); phases=rng.normal(0,0.08,m); slopes=rng.normal(0,0.15,m)
    clean=np.array([a*np.sin(2*np.pi*(t+p))+0.5*np.sin(4*np.pi*t)+s*(t-0.5) for a,p,s in zip(amps,phases,slopes)])
    noisy=clean+rng.normal(0,0.35,clean.shape)
    pca=PCA(n_components=4).fit(noisy)
    rec=pca.inverse_transform(pca.transform(noisy))
    idx=[2,10,20]
    fig,axs=plt.subplots(1,3,figsize=(10.0,3.3),sharey=True)
    for ax,i in zip(axs,idx):
        ax.plot(t,clean[i],color="black",lw=2,label="clean")
        ax.plot(t,noisy[i],color=COL["gray"],lw=1,alpha=0.7,label="noisy")
        ax.plot(t,rec[i],color=COL["red"],lw=2,label="PCA reconstruction")
        ax.set_title(f"sample {i}")
        ax.set_xlabel("normalized time")
        add_grid(ax); sanitize_for_ax(ax)
    axs[0].set_ylabel("signal")
    axs[0].legend(frameon=True,loc="lower left")
    fig.suptitle("Denoising by reconstructing from a low-dimensional PCA subspace",y=1.05)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_pca_preprocessing_accuracy(out_dir, base):
    X,Xs,y,_=wine_data()
    dims=list(range(1,Xs.shape[1]+1))
    scores=[]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    for r in dims:
        Z=PCA(n_components=r).fit_transform(Xs)
        clf=LogisticRegression(max_iter=2000,random_state=0)
        scores.append(cross_val_score(clf,Z,y,cv=cv).mean())
    full=cross_val_score(LogisticRegression(max_iter=2000,random_state=0),Xs,y,cv=cv).mean()
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    ax.plot(dims,scores,marker="o",lw=2.0,color=COL["blue"],label="PCA features")
    ax.axhline(full,color=COL["orange"],lw=2,ls="--",label="all standardized features")
    ax.set_ylim(0.75,1.02)
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title("PCA as preprocessing before classification")
    ax.legend(frameon=True); add_grid(ax); sanitize_for_ax(ax)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_tsne_local_neighbors(out_dir, base):
    _,Xs,y,_=wine_data()
    Zp=PCA(n_components=2).fit_transform(Xs)
    Zt=TSNE(n_components=2,perplexity=30,init="pca",learning_rate="auto",random_state=3,max_iter=750).fit_transform(Xs)
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.8))
    for ax,Z,title in [(axs[0],Zp,"Principal Component Analysis (PCA)"),(axs[1],Zt,"t-distributed Stochastic Neighbor Embedding (t-SNE)")]:
        scatter_classes(ax,Z,y,s=28)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(True)
    axs[0].legend(frameon=True,loc="best")
    fig.suptitle("PCA preserves global linear variance; t-SNE emphasizes local neighborhoods",y=1.05)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_tsne_wine_perplexity(out_dir, base):
    _,Xs,y,_=wine_data()
    perplexities=[5,30,50]
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.5))
    for ax,p in zip(axs,perplexities):
        Z=TSNE(n_components=2,perplexity=p,init="pca",learning_rate="auto",random_state=5,max_iter=750).fit_transform(Xs)
        scatter_classes(ax,Z,y,s=22,labels=False)
        ax.set_title(f"perplexity = {p}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(True)
    fig.suptitle("t-SNE layout changes with the neighborhood scale",y=1.05)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_isomap_swiss_roll(out_dir, base):
    X,t=make_swiss_roll(n_samples=800,noise=0.06,random_state=0)
    Z=Isomap(n_neighbors=10,n_components=2).fit_transform(X)
    fig=plt.figure(figsize=(9.0,4.0))
    ax1=fig.add_subplot(1,2,1,projection="3d")
    p=ax1.scatter(X[:,0],X[:,1],X[:,2],c=t,cmap="viridis",s=8,alpha=0.9)
    ax1.set_title("rolled surface in 3D")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.view_init(elev=12,azim=-65)
    ax2=fig.add_subplot(1,2,2)
    ax2.scatter(Z[:,0],Z[:,1],c=t,cmap="viridis",s=9,edgecolor="none")
    ax2.set_title("Isometric Mapping (Isomap) embedding")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_frame_on(True)
    fig.suptitle("Isomap uses graph distances to unfold a nonlinear manifold",y=1.03)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_method_comparison_wine(out_dir, base):
    _,Xs,y,_=wine_data()
    Zp=PCA(n_components=2).fit_transform(Xs)
    Zt=TSNE(n_components=2,perplexity=30,init="pca",learning_rate="auto",random_state=11,max_iter=750).fit_transform(Xs)
    Zi=Isomap(n_neighbors=10,n_components=2).fit_transform(Xs)
    methods=[("Principal Component Analysis (PCA)",Zp),("t-SNE",Zt),("Isometric Mapping (Isomap)",Zi)]
    fig,axs=plt.subplots(1,3,figsize=(11.0,3.6))
    for ax,(title,Z) in zip(axs,methods):
        scatter_classes(ax,Z,y,s=22,labels=False)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(True)
    fig.suptitle("The same Wine data summarized by three scikit-learn methods",y=1.05)
    fig.tight_layout(); save_figure(fig,out_dir,base)


def fig_dimred_workflow(out_dir, base):
    fig,ax=plt.subplots(figsize=(10.0,4.4))
    ax.set_xlim(-5.3,5.3); ax.set_ylim(-2.3,2.3); ax.axis("off")
    steps=[("raw data\n$X$",-4.4,0,"#d9edf7"),
           ("train/test\nsplit",-2.7,0,"#fff2cc"),
           ("fit scaler\non train",-1.0,0,"#fde2bd"),
           ("fit PCA\nor t-SNE",0.8,0,"#dff0d8"),
           ("inspect\nembedding",2.6,0,"#e6e0f8"),
           ("validate\ndownstream",4.4,0,"#f8d7da")]
    patches=[]
    for text,x,y,c in steps:
        patches.append(make_flow_box(ax,(x,y),text,c,width=1.45,height=0.85,fontsize=9))
    for (text1,x1,_,_),(text2,x2,_,_) in zip(steps[:-1],steps[1:]):
        arrow(ax,(x1+0.72,0),(x2-0.72,0),ms=10)
    ax.text(0,1.35,"Fit transformations on training data; apply the same transformation to new data",ha="center",fontsize=12,weight="bold")
    ax.text(-2.7,-1.25,"avoid leakage",ha="center",bbox=dict(facecolor="white",edgecolor=COL["orange"],boxstyle="round,pad=0.25"))
    ax.text(0.8,-1.25,"save fitted parameters",ha="center",bbox=dict(facecolor="white",edgecolor=COL["green"],boxstyle="round,pad=0.25"))
    ax.text(3.5,-1.25,"check the engineering task",ha="center",bbox=dict(facecolor="white",edgecolor=COL["red"],boxstyle="round,pad=0.25"))
    fig.tight_layout(); save_figure(fig,out_dir,base)


FIGURE_FUNCTIONS = {
    # Chapter 8
    "wine_data_feature-feature": fig_wine_data_feature_feature,
    "Clustering_vs_Classification": fig_clustering_vs_classification,
    "Clustering_methods": fig_clustering_methods,
    "K-means_wine_data": fig_kmeans_wine_data,
    "K-Means_decision_regions": fig_kmeans_decision_regions,
    "K-means_centroid_updates": fig_kmeans_centroid_updates,
    "K-Means_random_starts": fig_kmeans_random_starts,
    "K-Means_cluster_diagnostics": fig_kmeans_cluster_diagnostics,
    "K-Means_Image_segmentation": fig_kmeans_image_segmentation,
    "hierarchical_agglomerative_clustering": fig_hierarchical_agglomerative_clustering,
    "DBSCAN_vs_k-means": fig_dbscan_vs_k_means,
    "DBSCAN_eps_too_small": fig_dbscan_eps_too_small,
    "DBSCAN_eps_neighborhoods": fig_dbscan_eps_neighborhoods,
    "Gaussian_general_idea": fig_gaussian_general_idea,
    "Gaussian_mixture_model": fig_gaussian_mixture_model,
    "Gaussian_mixture_cluster": fig_gaussian_mixture_cluster,
    "Gaussian_mixture_covariance": fig_gaussian_mixture_covariance,
    "Gaussian_anomaly_detection": fig_gaussian_anomaly_detection,
    # Chapter 9
    "DimRed_overview": fig_dimred_overview,
    "DimRed_wine_feature_matrix": fig_dimred_wine_feature_matrix,
    "DimRed_pca_shadow_projection": fig_dimred_pca_shadow_projection,
    "DimRed_pca_projection_geometry": fig_dimred_pca_projection_geometry,
    "DimRed_covariance_matrix_structure": fig_dimred_covariance_matrix_structure,
    "DimRed_wine_covariance_heatmap": fig_dimred_wine_covariance_heatmap,
    "DimRed_wine_scree_plot": fig_dimred_wine_scree_plot,
    "DimRed_wine_pca_loadings": fig_dimred_wine_pca_loadings,
    "DimRed_wine_pca_scores": fig_dimred_wine_pca_scores,
    "DimRed_pca_reconstruction_error": fig_dimred_pca_reconstruction_error,
    "DimRed_pca_reconstruction_denoising": fig_dimred_pca_reconstruction_denoising,
    "DimRed_pca_preprocessing_accuracy": fig_dimred_pca_preprocessing_accuracy,
    "DimRed_tsne_local_neighbors": fig_dimred_tsne_local_neighbors,
    "DimRed_tsne_wine_perplexity": fig_dimred_tsne_wine_perplexity,
    "DimRed_isomap_swiss_roll": fig_dimred_isomap_swiss_roll,
    "DimRed_method_comparison_wine": fig_dimred_method_comparison_wine,
    "DimRed_workflow": fig_dimred_workflow,
}


def make_figure(base_name: str, out_dir: str | Path = "."):
    set_style()
    if base_name not in FIGURE_FUNCTIONS:
        raise ValueError(f"Unknown figure name: {base_name}. Known names: {sorted(FIGURE_FUNCTIONS)}")
    FIGURE_FUNCTIONS[base_name](out_dir, base_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate one course figure.")
    parser.add_argument("name", help="figure base name")
    parser.add_argument("--out", default=".", help="output folder")
    args = parser.parse_args()
    make_figure(args.name, args.out)
