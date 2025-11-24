import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

plt.close('all')

# === Global Font Settings ===
fontSize = 8
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': fontSize,
    'axes.titlesize': fontSize,
    'axes.labelsize': fontSize,
    'xtick.labelsize': fontSize,
    'ytick.labelsize': fontSize,
    'legend.fontsize': fontSize,
    'figure.titlesize': fontSize
})

# === SETTINGS ===
line_width = 1.5
fig_width_in = 7
fig_height_in = 3.5

# === File paths ===
save_folder = os.path.join(
    r"C:\Users\JWORCH\OneDrive - cecsc\Documents - Downy-Khan-research-group",
    "Publications",
    "Conference_publications",
    "2026",
    "Worch-ITEC_Digest",
    "LaTeX",
    "Figures",
)
os.makedirs(save_folder, exist_ok=True)

# === Generate circular dataset ===
np.random.seed(0)
X = np.random.uniform(-1, 1, (200, 2))
y = np.where(X[:, 0]**2 + X[:, 1]**2 > 0.5, 1, -1)

# === Explicit polynomial feature map: (x1^2, sqrt(2)x1x2, x2^2) ===
phi = np.c_[X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2]

# Train a *linear* SVM in feature space to get a true separating plane
clf_linear = SVC(kernel="linear", C=10)
clf_linear.fit(phi, y)
w = clf_linear.coef_[0]
b = clf_linear.intercept_[0]

# === Build figure ===
fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=200)

# Left panel: original 2D data
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='+', s=20)
ax1.scatter(X[y == -1, 0], X[y == -1, 1], facecolors='none', edgecolors='b', s=20)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_title("2D Data (not linearly separable)")

# Right panel: 3D feature space
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Data points
ax2.scatter(phi[y == 1, 0], phi[y == 1, 1], phi[y == 1, 2],
            c='r', marker='+', s=20, depthshade=True)
ax2.scatter(phi[y == -1, 0], phi[y == -1, 1], phi[y == -1, 2],
            facecolors='none', edgecolors='b', s=20, depthshade=True)

# Parabola surface
x1 = np.linspace(-1, 1, 18)
x2 = np.linspace(-1, 1, 18)
X1, X2 = np.meshgrid(x1, x2)

Y1 = X1**2
Y2 = np.sqrt(2) * X1 * X2 + 0.05
Y3 = X2**2

ax2.plot_surface(Y1, Y2, Y3,
                 color="white",
                 edgecolor="k",
                 linewidth=0.8,
                 alpha=0.9,
                 shade=False)

# Separating plane
xx, yy = np.meshgrid(np.linspace(0, 1, 20),
                     np.linspace(-1.5, 1.5, 20))
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

ax2.plot_surface(xx, yy, zz,
                 color="gray",
                 alpha=0.95,
                 edgecolor="none")

# Fix axis bounds
ax2.set_xlim(0, 1)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(0, 1)

ax2.set_title("Feature Space (linearly separable)")
ax2.view_init(elev=20, azim=-110, roll=0)

plt.suptitle("Nonlinear SVM Kernel Mapping (Polynomial Degree 2)", fontsize=fontSize)

plt.tight_layout()

#%% === Save Figure ===
# Get current Python filename without extension
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Build initial save path
base_name = script_name + ".png"
fig_path = os.path.join(save_folder, base_name)

# Increment filename if it already exists
counter = 1
while os.path.exists(fig_path):
    new_name = f"{script_name}_{counter}.png"
    fig_path = os.path.join(save_folder, new_name)
    counter += 1

# Save the figure
fig.savefig(fig_path, dpi=500, bbox_inches="tight")
print(f"Saved Figure to: {fig_path}")
