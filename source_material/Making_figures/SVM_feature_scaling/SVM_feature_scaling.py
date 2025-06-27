import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# — Plot formatting (Austin Downey default) —
plt.rcParams.update({
    'text.usetex': True,
    'image.cmap': 'viridis',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                   'Bitstream Vera Serif', 'Computer Modern Roman',
                   'New Century Schoolbook', 'Century Schoolbook L', 'Utopia',
                   'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
                   'Charter', 'serif'],
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.rm': 'serif',
    'mathtext.fontset': 'custom'
})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.close('all')

# ----------------------------------------------------------
#   Helper – only computes, no plotting
# ----------------------------------------------------------
def compute_svc_boundary(svm_clf, xmin, xmax, n=200):
    """Return boundary and margin lines plus support vectors."""
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, n)

    decision = -w[0] / w[1] * x0 - b / w[1]
    margin    = 1 / w[1]
    gutter_up = decision + margin
    gutter_dn = decision - margin
    svs = svm_clf.support_vectors_

    return x0, decision, gutter_up, gutter_dn, svs

# ----------------------------------------------------------
#   Example 1 – Iris (large-margin classification)
# ----------------------------------------------------------
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]     # petal length & width
y = iris["target"]
mask = (y == 0) | (y == 1)      # setosa vs. versicolor
X, y = X[mask], y[mask]

svm_clf_iris = SVC(kernel="linear", C=10_000).fit(X, y)

# ----------------------------------------------------------
#   Example 2 – Sensitivity to feature scales
# ----------------------------------------------------------
Xs  = np.array([[1, 50], [5, 20], [3, 80], [5, 60]], dtype=float)
ys  = np.array([0, 0, 1, 1])

clf_raw    = SVC(kernel="linear", C=100).fit(Xs, ys)
x0_raw, dec_raw, up_raw, dn_raw, sv_raw = compute_svc_boundary(clf_raw, 0, 6)

scaler     = StandardScaler()
Xs_scaled  = scaler.fit_transform(Xs)
clf_scaled = SVC(kernel="linear", C=100).fit(Xs_scaled, ys)
x0_scl, dec_scl, up_scl, dn_scl, sv_scl = compute_svc_boundary(clf_scaled, -2, 2)

# ----------------------------------------------------------
#   Plotting (subplot style)
# ----------------------------------------------------------
plt.figure(figsize=(6.5, 2.5))

# ── Unscaled ──────────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.scatter(Xs[ys==1, 0], Xs[ys==1, 1], marker='o',zorder=10)
plt.scatter(Xs[ys==0, 0], Xs[ys==0, 1], marker='s',zorder=10)
plt.scatter(sv_raw[:, 0], sv_raw[:, 1],  s=150, facecolors="none",
            edgecolors=cc[3], linewidths=2,zorder=10)
plt.plot(x0_raw, dec_raw, 'k-',  lw=2)
plt.plot(x0_raw, up_raw,  'k--', lw=2)
plt.plot(x0_raw, dn_raw,  'k--', lw=2)
plt.xlabel(r"$x_0$")
plt.ylabel(r"$x_1$")
plt.title("unscaled")
plt.axis([0, 6, 0, 90])

# ── Scaled ────────────────────────────────────────────────
plt.subplot(1, 2, 2)
plt.plot(x0_scl, dec_scl, 'k-',  lw=2)
plt.plot(x0_scl, up_scl,  'k--', lw=2)
plt.plot(x0_scl, dn_scl,  'k--', lw=2)
plt.scatter(Xs_scaled[ys==1, 0], Xs_scaled[ys==1, 1], marker='o',zorder=10)
plt.scatter(Xs_scaled[ys==0, 0], Xs_scaled[ys==0, 1], marker='s',zorder=10)
plt.scatter(sv_scl[:, 0], sv_scl[:, 1],  s=150, facecolors="none",
            edgecolors=cc[3], linewidths=2,zorder=10)

plt.xlabel(r"$x'_0$")
plt.ylabel(r"$x'_1$")
plt.title("scaled")
plt.axis([-2, 2, -2, 2])

plt.tight_layout()
plt.savefig("SVM_feature_scaling", dpi=300)












