"""
Neural Network Visual Explainer
================================
Interactive Streamlit app that animates how a neural network learns.
Designed for school students — no ML background needed.

Run:  streamlit run neural_network_animation.py
Deps: pip install streamlit matplotlib numpy Pillow
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Network Explorer", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body { background-color: #0a0a1a; }
.step-card {
    background: #12122a;
    border-left: 5px solid #FFD93D;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #eee;
}
.stButton > button {
    background: linear-gradient(135deg, #FF6B6B, #FFD93D);
    color: white; font-weight: 700; border: none;
    border-radius: 10px; padding: 0.4rem 1.4rem; font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Architecture ───────────────────────────────────────────────────────────────
LAYER_SIZES = [3, 4, 4, 2]
LAYER_NAMES = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
LAYER_X     = [0.10, 0.37, 0.63, 0.90]
NODE_COLORS = ["#4CC9F0", "#7209B7", "#7209B7", "#F72585"]
NODE_R      = 0.038

SAMPLES = {
    "Cat 🐱": np.array([0.9, 0.2, 0.7]),
    "Dog 🐶": np.array([0.3, 0.8, 0.5]),
    "Car 🚗": np.array([0.1, 0.4, 0.9]),
}

STEPS = [
    ("🖼️  Input",           "An image is converted to numbers (pixel values 0-1). Each number feeds into one input neuron."),
    ("➡️  Forward Pass",     "Signals travel layer by layer. Each connection has a weight that scales the signal. Yellow = active signal."),
    ("⚡  Activation (ReLU)","Each neuron applies ReLU: if the sum is negative, set to 0. This lets the network learn complex patterns."),
    ("🏁  Output",           "The final layer gives probabilities for each class. The highest value is the network's prediction."),
    ("❌  Loss",             "We compare the prediction to the correct answer. The gap is the Loss — how wrong the network was."),
    ("🔙  Backpropagation",  "The error travels backwards through the network. Each weight learns how much it contributed to the mistake."),
    ("🔧  Weight Update",    "Weights are nudged to reduce the error (gradient descent). Repeat many times and the network learns! Triangle = weight change."),
]

# ── Maths ──────────────────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0.0, x)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def forward(inputs, weights):
    acts = [np.array(inputs, dtype=float)]
    cur  = acts[0].copy()
    for i, W in enumerate(weights):
        z   = W @ cur
        cur = relu(z) if i < len(weights) - 1 else softmax(z)
        acts.append(cur)
    return acts

def init_weights():
    np.random.seed(42)
    return [
        np.random.randn(LAYER_SIZES[i+1], LAYER_SIZES[i]) * 0.6
        for i in range(len(LAYER_SIZES) - 1)
    ]

def compute_grads(activations, weights, target):
    delta = 2.0 * (activations[-1] - target)
    grads = []
    for li in range(len(weights) - 1, -1, -1):
        dW = np.outer(delta, activations[li])
        grads.insert(0, dW)
        delta = (weights[li].T @ delta) * (activations[li] > 0)
    return grads

# ── Node positions ─────────────────────────────────────────────────────────────
def node_pos():
    pos = {}
    for li, (lx, n) in enumerate(zip(LAYER_X, LAYER_SIZES)):
        ys = np.linspace(0.15, 0.85, n)
        for ni, y in enumerate(ys):
            pos[(li, ni)] = (lx, float(y))
    return pos

POS = node_pos()

# ── Drawing ────────────────────────────────────────────────────────────────────
def hex_brighten(hex_color, val):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r2 = int(r + (255 - r) * val)
    g2 = int(g + (255 - g) * val)
    b2 = int(b + (255 - b) * val)
    return f"#{r2:02x}{g2:02x}{b2:02x}"

def draw_network(activations, weights, step, grads=None):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layer labels
    for li, (lx, name) in enumerate(zip(LAYER_X, LAYER_NAMES)):
        ax.text(lx, 0.04, name, ha="center", va="bottom",
                fontsize=8.5, color="#aaaaaa", fontweight="bold", fontfamily="monospace")

    # Edges
    for li in range(len(LAYER_SIZES) - 1):
        for ni in range(LAYER_SIZES[li]):
            for nj in range(LAYER_SIZES[li + 1]):
                x1, y1 = POS[(li,     ni)]
                x2, y2 = POS[(li + 1, nj)]
                w      = weights[li][nj, ni]

                color = "#4CC9F0" if w >= 0 else "#FF6B6B"
                lw    = 0.5
                alpha = 0.15

                # forward pass glow
                if step in (1, 2, 3) and li <= step - 1:
                    a_val = float(np.clip(activations[li][ni], 0, 1))
                    alpha = max(0.05, a_val * 0.85)
                    lw    = 0.4 + 2.5 * a_val
                    color = "#FFD93D"

                # backprop highlight
                if step == 5 and li == 2:
                    color = "#F72585"; lw = 1.8; alpha = 0.7
                if step == 6 and li == 1:
                    color = "#F72585"; lw = 1.8; alpha = 0.7

                ax.plot([x1, x2], [y1, y2],
                        color=color, lw=lw, alpha=alpha, zorder=1)

    # Nodes
    for (li, ni), (x, y) in POS.items():
        base  = NODE_COLORS[li]
        a_val = float(np.clip(activations[li][ni], 0, 1)) if ni < len(activations[li]) else 0.0
        fill  = hex_brighten(base, a_val * 0.8) if step > 0 else base

        # glow ring — no transform= argument
        glow = plt.Circle((x, y), NODE_R * 1.65, color=fill, alpha=0.18, zorder=2)
        ax.add_patch(glow)

        # main node — no transform= argument
        node = plt.Circle((x, y), NODE_R, color=fill, zorder=3,
                           linewidth=1.2, edgecolor="white")
        ax.add_patch(node)

        # activation value
        if step > 0 and ni < len(activations[li]):
            ax.text(x, y, f"{activations[li][ni]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold", zorder=5)

        # weight-update indicators
        if step == 6 and grads is not None and li > 0:
            gi = li - 1
            if gi < len(grads) and ni < grads[gi].shape[0]:
                d   = float(grads[gi][ni].mean())
                sign = "^" if d < 0 else "v"
                col  = "#00FF99" if d < 0 else "#FF6B6B"
                ax.text(x + 0.048, y + 0.03, sign,
                        fontsize=9, color=col, zorder=6, fontweight="bold")

    # Title
    ax.text(0.5, 0.97, STEPS[step][0],
            ha="center", va="top", fontsize=13,
            fontweight="bold", color="white", fontfamily="monospace")

    plt.tight_layout(pad=0.3)
    return fig

# ── Session state ──────────────────────────────────────────────────────────────
if "step"         not in st.session_state: st.session_state.step         = 0
if "weights"      not in st.session_state: st.session_state.weights      = init_weights()
if "loss_history" not in st.session_state: st.session_state.loss_history = []
if "epoch"        not in st.session_state: st.session_state.epoch        = 0
if "sample"       not in st.session_state: st.session_state.sample       = "Cat 🐱"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Neural Network Explorer")
    st.markdown("*A visual guide for curious minds!*")
    st.divider()

    chosen = st.selectbox("Choose an input:", list(SAMPLES.keys()),
                          index=list(SAMPLES.keys()).index(st.session_state.sample))
    if chosen != st.session_state.sample:
        st.session_state.sample = chosen

    st.divider()
    st.markdown("### Architecture")
    st.markdown(f"- **Inputs:** {LAYER_SIZES[0]} neurons\n- **Hidden:** 2 x {LAYER_SIZES[1]} neurons\n- **Output:** {LAYER_SIZES[-1]} neurons")

    st.divider()
    st.markdown("### Legend")
    st.markdown("""
Yellow edge = active signal  
Blue edge = positive weight  
Red edge = negative weight  
Pink edge = backprop gradient  
Green ^ = weight will increase  
Red v = weight will decrease  
""")

    st.divider()
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)

    if st.button("Reset Network"):
        st.session_state.weights      = init_weights()
        st.session_state.loss_history = []
        st.session_state.epoch        = 0
        st.session_state.step         = 0
        st.rerun()

# ── Compute ────────────────────────────────────────────────────────────────────
inputs      = SAMPLES[st.session_state.sample]
weights     = st.session_state.weights
activations = forward(inputs, weights)
step        = st.session_state.step

target = np.array([1.0, 0.0]) if "Cat" in st.session_state.sample else np.array([0.0, 1.0])
pred   = activations[-1]
loss   = float(np.mean((pred - target) ** 2))
grads  = compute_grads(activations, weights, target)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧠 How Does a Neural Network Learn?")
st.caption("Step through each phase to see what is happening inside the network.")

# ── Navigation ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 5, 1])
with c1:
    if st.button("Prev") and step > 0:
        st.session_state.step -= 1
        st.rerun()
with c3:
    if st.button("Next") and step < len(STEPS) - 1:
        st.session_state.step += 1
        if st.session_state.step == 6:
            for li in range(len(weights)):
                weights[li] -= lr * grads[li]
            st.session_state.weights      = weights
            st.session_state.epoch       += 1
            st.session_state.loss_history.append(loss)
        st.rerun()

step = st.session_state.step
st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)}")

# ── Main layout ────────────────────────────────────────────────────────────────
col_fig, col_info = st.columns([2, 1])

with col_fig:
    fig = draw_network(activations, weights, step,
                       grads=grads if step == 6 else None)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_info:
    title, desc = STEPS[step]
    st.markdown(f"""
<div class="step-card">
  <h3 style="color:#FFD93D;margin:0 0 0.5rem 0">{title}</h3>
  <p style="margin:0;line-height:1.65;color:#ddd">{desc}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Predictions")
    labels = ["Cat 🐱", "Not Cat"]
    for label, prob in zip(labels, pred):
        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    st.markdown(f"**Loss:** `{loss:.4f}`")
    st.markdown(f"**Epoch:** `{st.session_state.epoch}`")

    if step == 0:
        st.markdown("#### Input Values")
        for name, val in zip(["Pixel R", "Pixel G", "Pixel B"], inputs):
            st.progress(float(val), text=f"{name}: {val:.2f}")

# ── Loss curve ─────────────────────────────────────────────────────────────────
if st.session_state.loss_history:
    st.divider()
    st.markdown("### Loss Curve — The Network is Learning!")

    lh = np.array(st.session_state.loss_history)
    fig2, ax2 = plt.subplots(figsize=(9, 2.8), facecolor="#0a0a1a")
    ax2.set_facecolor("#0a0a1a")
    ax2.plot(lh, color="#FFD93D", lw=2.5, marker="o", markersize=5)
    ax2.fill_between(range(len(lh)), lh, alpha=0.18, color="#FFD93D")
    ax2.set_xlabel("Epoch", color="#999")
    ax2.set_ylabel("Loss", color="#999")
    ax2.tick_params(colors="#999")
    for spine in ax2.spines.values():
        spine.set_color("#333")
    ax2.set_title("As loss decreases, predictions get better!", color="white", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Auto-play ──────────────────────────────────────────────────────────────────
st.divider()
ac1, ac2 = st.columns([1, 3])
with ac1:
    auto = st.toggle("Auto-play")
with ac2:
    speed = st.slider("Seconds per step", 0.5, 3.0, 1.5, 0.5)

if auto:
    time.sleep(speed)
    if step < len(STEPS) - 1:
        st.session_state.step += 1
        if st.session_state.step == 6:
            for li in range(len(weights)):
                weights[li] -= lr * grads[li]
            st.session_state.weights      = weights
            st.session_state.epoch       += 1
            st.session_state.loss_history.append(loss)
    else:
        st.session_state.step = 0
    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#555;font-size:0.85rem'>"
    "Built with love for curious students"
    "</center>",
    unsafe_allow_html=True,
)
