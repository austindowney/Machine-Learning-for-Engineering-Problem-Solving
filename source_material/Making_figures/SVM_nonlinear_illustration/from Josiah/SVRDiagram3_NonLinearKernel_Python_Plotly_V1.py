import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
import plotly.io as pio

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

# === Create subplot layout (1 row, 2 columns) ===
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy"}, {"type": "scene"}]],
    #subplot_titles=("2D Data (not linearly separable)",
                    #"Feature Space (linearly separable)")
)

# === Left panel: original 2D data ===
fig.add_trace(
    go.Scatter(
        x=X[y == 1, 0], y=X[y == 1, 1],
        mode="markers", marker=dict(color="red", symbol="cross", size=6),
        name="+1 (2D)"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=X[y == -1, 0], y=X[y == -1, 1],
        mode="markers", marker=dict(color="blue", symbol="circle-open", size=6),
        name="-1 (2D)"
    ),
    row=1, col=1
)

# === Right panel: 3D feature space ===
# Data points
fig.add_trace(
    go.Scatter3d(
        x=phi[y == 1, 0], y=phi[y == 1, 1], z=phi[y == 1, 2],
        mode="markers", marker=dict(color="red", symbol="cross", size=4),
        name="+1 (3D)"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=phi[y == -1, 0], y=phi[y == -1, 1], z=phi[y == -1, 2],
        mode="markers", marker=dict(color="blue", symbol="circle-open", size=4),
        name="-1 (3D)"
    ),
    row=1, col=2
)

# Parabola surface (solid white with black contours)
x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)
X1, X2 = np.meshgrid(x1, x2)

Y1 = X1**2
Y2 = np.sqrt(2) * X1 * X2 + 0.05  # slight shift upward
Y3 = X2**2

fig.add_trace(
    go.Surface(
        x=Y1, y=Y2, z=Y3,
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        opacity=1.0,
        contours=dict(
            x=dict(show=True, color="black", width=1),
            y=dict(show=True, color="black", width=1),
            z=dict(show=True, color="black", width=1)
        ),
        #name="Feature mapping surface"
    ),
    row=1, col=2
)

# === Separating plane ===
xx, yy = np.meshgrid(np.linspace(0, 1, 20),
                     np.linspace(-1.5, 1.5, 20))
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

fig.add_trace(
    go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, "gray"], [1, "gray"]],
        showscale=False,
        opacity=0.95,
        name="Separating plane"
    ),
    row=1, col=2
)

# === Layout settings ===
fig.update_xaxes(range=[-1, 1], row=1, col=1)
fig.update_yaxes(range=[-1, 1], row=1, col=1)

fig.update_layout(
    #title_text="Nonlinear SVM Kernel Mapping (Polynomial Degree 2)",
    scene=dict(
        #xaxis_title="x₁²",
        #yaxis_title="√2·x₁x₂",
        #zaxis_title="x₂²",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-1.5, 1.5]),
        zaxis=dict(range=[0, 1]),
        aspectmode="manual",              # force equal aspect
        aspectratio=dict(x=1, y=1, z=1),  # cube aspect ratio
        camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9))
    ),
    height=600,
    width=1100,
    showlegend=False
)

# Render in browser
import plotly.io as pio
pio.renderers.default = "browser"
fig.show()
