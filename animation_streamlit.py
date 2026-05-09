
"""
Neural Network Visual Explainer
================================
A Streamlit app that animates how a neural network works —
forward pass (signal flow), activation, and backpropagation
(weight updates). Designed for school students in rural India.

Run with:  streamlit run neural_network_animation.py
Requirements: pip install streamlit matplotlib numpy Pillow
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;700&family=Nunito:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
    h1, h2, h3 { font-family: 'Baloo 2', cursive; }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #FFD93D);
        color: white; font-weight: 700; border: none;
        border-radius: 12px; padding: 0.5rem 1.5rem;
        font-size: 1rem; cursor: pointer; transition: 0.3s;
    }
    .stButton > button:hover { opacity: 0.85; transform: scale(1.03); }
    .step-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 5px solid #FFD93D;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin: 0.5rem 0; color: #eee;
    }
    .legend-box {
        background: #0f3460; border-radius: 10px;
        padding: 0.8rem 1rem; color: #eee; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_SIZES  = [3, 4, 4, 2]          # input, hidden1, hidden2, output
LAYER_NAMES  = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
LAYER_X      = [0.12, 0.38, 0.62, 0.88]
NODE_COLORS  = ["#4CC9F0", "#7209B7", "#7209B7", "#F72585"]
EDGE_ALPHA   = 0.15
FIG_SIZE     = (13, 7)

SAMPLE_IMAGES = {
    "Cat 🐱":  np.array([0.9, 0.2, 0.7]),
    "Dog 🐶":  np.array([0.3, 0.8, 0.5]),
    "Car 🚗":  np.array([0.1, 0.4, 0.9]),
}

STEP_DESCRIPTIONS = [
    ("🖼️ Step 1 – Input", "An image is broken into numbers (pixel values 0–1). Each number feeds into one input neuron."),
    ("➡️ Step 2 – Forward Pass", "Signals travel forward. Each connection has a **weight** that multiplies the signal. Neurons add all incoming signals."),
    ("⚡ Step 3 – Activation", "Each neuron squishes its total through a ReLU function: if the value is negative, set to 0. This adds non-linearity!"),
    ("🏁 Step 4 – Output", "The final layer gives probabilities. The highest value = the network's prediction."),
    ("❌ Step 5 – Error (Loss)", "We compare the prediction to the **correct answer**. The difference is the Loss (error)."),
    ("🔙 Step 6 – Backpropagation", "The error flows *backwards*. Each weight learns how much it contributed to the mistake."),
    ("🔧 Step 7 – Weight Update", "Weights are nudged in the direction that reduces error (gradient descent). Repeat thousands of times → the network learns!"),
]

# ── Helper: draw the network ──────────────────────────────────────────────────

def node_positions():
    """Return dict {(layer, node): (x, y)} for all nodes."""
    pos = {}
    for li, (lx, n) in enumerate(zip(LAYER_X, LAYER_SIZES)):
        ys = np.linspace(0.15, 0.85, n)
        for ni, y in enumerate(ys):
            pos[(li, ni)] = (lx, y)
    return pos

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def forward_pass(inputs, weights):
    activations = [inputs]
    current = inputs
    for W in weights:
        z = W @ current
        a = relu(z)
        activations.append(a)
    # soft-max on last layer for probabilities
    last = activations[-1]
    exp = np.exp(last - last.max())
    activations[-1] = exp / exp.sum()
    return activations

def init_weights():
    np.random.seed(42)
    ws = []
    for i in range(len(LAYER_SIZES) - 1):
        W = np.random.randn(LAYER_SIZES[i+1], LAYER_SIZES[i]) * 0.5
        ws.append(W)
    return ws

def draw_network(
    pos, weights,
    activations=None,
    highlight_layer=None,
    highlight_edges_from=None,
    back_layer=None,
    weight_delta=None,
    title="Neural Network",
):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Layer labels
    for li, (lx, name) in enumerate(zip(LAYER_X, LAYER_NAMES)):
        ax.text(lx, 0.04, name, ha="center", va="center",
                fontsize=9, color="#aaa", fontweight="bold",
                fontfamily="monospace")

    # Draw edges
    for li in range(len(LAYER_SIZES) - 1):
        for ni in range(LAYER_SIZES[li]):
            for nj in range(LAYER_SIZES[li+1]):
                x1, y1 = pos[(li,   ni)]
                x2, y2 = pos[(li+1, nj)]

                w = weights[li][nj, ni]
                color = "#FF6B6B" if w < 0 else "#4CC9F0"
                lw    = min(abs(w) * 1.5 + 0.3, 2.5)
                alpha = EDGE_ALPHA

                # Highlight forward flow
                if highlight_edges_from == li:
                    if activations is not None:
                        a_val = float(activations[li][ni])
                        alpha = max(0.05, min(0.9, a_val))
                        lw    = 0.5 + 2.5 * alpha
                        color = "#FFD93D"

                # Highlight backprop
                if back_layer == li:
                    color = "#F72585"
                    lw    = 1.5
                    alpha = 0.6

                ax.plot([x1, x2], [y1, y2],
                        color=color, lw=lw, alpha=alpha, zorder=1)

    # Draw nodes
    for (li, ni), (x, y) in pos.items():
        base_color = NODE_COLORS[li]
        radius = 0.030

        # Activation brightness
        fill_color = base_color
        if activations is not None and li < len(activations):
            a_arr = activations[li]
            if ni < len(a_arr):
                v = float(a_arr[ni])
                # blend white into base color based on activation
                r, g, b = int(base_color[1:3],16), int(base_color[3:5],16), int(base_color[5:7],16)
                r2 = int(r + (255-r)*v); g2 = int(g + (255-g)*v); b2 = int(b + (255-b)*v)
                fill_color = f"#{r2:02x}{g2:02x}{b2:02x}"

        glow_radius = radius * 1.6
        if highlight_layer == li or (activations is not None and li < len(activations)):
            glow = plt.Circle((x, y), glow_radius, color=fill_color,
                              alpha=0.18, zorder=2, transform=ax.transData)
            ax.add_patch(glow)

        circle = plt.Circle((x, y), radius, color=fill_color,
                             zorder=3, transform=ax.transData,
                             linewidth=1.5, edgecolor="white")
        ax.add_patch(circle)

        # Show activation value
        if activations is not None and li < len(activations):
            a_arr = activations[li]
            if ni < len(a_arr):
                v = float(a_arr[ni])
                ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=5)

        # Weight delta indicator (backprop)
        if weight_delta is not None and back_layer is not None:
            if li == back_layer and li < len(weight_delta):
                dw = weight_delta[li]
                if ni < dw.shape[0]:
                    d = float(dw[ni].mean())
                    sign = "▲" if d > 0 else "▼"
                    ax.text(x + 0.035, y + 0.025, sign,
                            fontsize=7, color="#FFD93D", zorder=6)

    # Title
    ax.text(0.5, 0.97, title, ha="center", va="top",
            fontsize=13, fontweight="bold", color="white",
            fontfamily="monospace")

    plt.tight_layout(pad=0.2)
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step      = 0
if "weights" not in st.session_state:
    st.session_state.weights   = init_weights()
if "activations" not in st.session_state:
    st.session_state.activations = None
if "sample" not in st.session_state:
    st.session_state.sample    = "Cat 🐱"
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "epoch" not in st.session_state:
    st.session_state.epoch     = 0

pos     = node_positions()
weights = st.session_state.weights
step    = st.session_state.step

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Neural Network Explorer")
    st.markdown("*A visual guide for curious minds!*")
    st.divider()

    chosen = st.selectbox("Choose an input image:", list(SAMPLE_IMAGES.keys()),
                          index=list(SAMPLE_IMAGES.keys()).index(st.session_state.sample))
    st.session_state.sample = chosen

    st.divider()
    st.markdown("### Network Architecture")
    st.markdown(f"- **Input neurons:** {LAYER_SIZES[0]}")
    st.markdown(f"- **Hidden layers:** 2 × {LAYER_SIZES[1]} neurons")
    st.markdown(f"- **Output neurons:** {LAYER_SIZES[3]} (classes)")

    st.divider()
    st.markdown("### Legend")
    st.markdown("""
    <div class='legend-box'>
    🔵 <b>Blue edges</b> = positive weights<br>
    🔴 <b>Red edges</b> = negative weights<br>
    🟡 <b>Yellow glow</b> = active signal<br>
    🌸 <b>Pink edges</b> = backprop gradient<br>
    ▲▼ = weight increasing / decreasing
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
    if st.button("🔄 Reset Network"):
        st.session_state.weights      = init_weights()
        st.session_state.activations  = None
        st.session_state.step         = 0
        st.session_state.loss_history = []
        st.session_state.epoch        = 0
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 How Does a Neural Network Learn?")
st.caption("Step through each phase to see what's happening inside the network.")

inputs = SAMPLE_IMAGES[st.session_state.sample]

# Compute activations if not done
if st.session_state.activations is None:
    st.session_state.activations = forward_pass(inputs, weights)

activations = st.session_state.activations

# Target: one-hot (first class = correct for Cat, second for others)
target = np.array([1.0, 0.0]) if "Cat" in chosen else np.array([0.0, 1.0])
pred   = activations[-1]
loss   = float(np.mean((pred - target) ** 2))

# Compute weight gradients (simple MSE backprop approximation for display)
delta = 2 * (pred - target)
weight_delta = []
for li in range(len(weights)):
    dw = np.outer(delta, activations[li]) * 0.1
    weight_delta.append(dw)
    delta = weights[li].T @ delta
    delta = (delta > 0).astype(float) * delta  # ReLU deriv

# ── Step navigation ───────────────────────────────────────────────────────────
col_prev, col_step, col_next = st.columns([1, 6, 1])
with col_prev:
    if st.button("◀ Prev") and step > 0:
        st.session_state.step -= 1
        st.rerun()
with col_next:
    if st.button("Next ▶") and step < len(STEP_DESCRIPTIONS) - 1:
        st.session_state.step += 1
        # On backprop step, update weights slightly
        if st.session_state.step == 6:
            for li in range(len(weights)):
                weights[li] -= lr * weight_delta[li]
            st.session_state.weights   = weights
            st.session_state.epoch    += 1
            st.session_state.loss_history.append(loss)
            st.session_state.activations = forward_pass(inputs, weights)
        else:
            st.session_state.activations = forward_pass(inputs, weights)
        st.rerun()

step = st.session_state.step
step_title, step_desc = STEP_DESCRIPTIONS[step]

# Step progress bar
st.progress((step + 1) / len(STEP_DESCRIPTIONS),
            text=f"Step {step+1} of {len(STEP_DESCRIPTIONS)}")

# ── Determine drawing parameters per step ────────────────────────────────────
highlight_layer       = None
highlight_edges_from  = None
back_layer            = None
wd_display            = None
net_title             = step_title.replace("\n", " ")

if step == 0:                        # Input
    highlight_layer = 0
elif step == 1:                      # Forward pass L0→L1
    highlight_edges_from = 0
elif step == 2:                      # Activation L1
    highlight_layer = 1
    highlight_edges_from = 1
elif step == 3:                      # Output
    highlight_layer = 3
    highlight_edges_from = 2
elif step == 4:                      # Loss
    highlight_layer = 3
elif step == 5:                      # Backprop
    back_layer = 2
elif step == 6:                      # Weight update
    back_layer    = 1
    wd_display    = weight_delta

# ── Draw ──────────────────────────────────────────────────────────────────────
fig = draw_network(
    pos, weights,
    activations          = activations,
    highlight_layer      = highlight_layer,
    highlight_edges_from = highlight_edges_from,
    back_layer           = back_layer,
    weight_delta         = wd_display,
    title                = net_title,
)

col_net, col_info = st.columns([2, 1])

with col_net:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_info:
    st.markdown(f"""
    <div class='step-box'>
    <h3 style='color:#FFD93D; margin:0 0 0.5rem 0'>{step_title}</h3>
    <p style='margin:0; line-height:1.6'>{step_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Current Predictions")
    output_labels = ["Cat 🐱", "Not Cat"]
    for label, prob in zip(output_labels, pred):
        st.metric(label=label, value=f"{prob*100:.1f}%")

    st.markdown(f"**Loss (error):** `{loss:.4f}`")
    st.markdown(f"**Training epoch:** `{st.session_state.epoch}`")

    if step == 0:
        st.markdown("#### 🖼️ Input Values")
        pnames = ["Pixel R", "Pixel G", "Pixel B"]
        for pn, pv in zip(pnames, inputs):
            st.progress(float(pv), text=f"{pn}: {pv:.2f}")

# ── Loss curve (shows after first backprop) ───────────────────────────────────
if st.session_state.loss_history:
    st.divider()
    st.markdown("### 📉 Loss Over Epochs (Network Learning!)")
    loss_arr = np.array(st.session_state.loss_history)
    fig2, ax2 = plt.subplots(figsize=(8, 2.5), facecolor="#0a0a1a")
    ax2.set_facecolor("#0a0a1a")
    ax2.plot(loss_arr, color="#FFD93D", lw=2.5, marker="o", markersize=5)
    ax2.fill_between(range(len(loss_arr)), loss_arr, alpha=0.15, color="#FFD93D")
    ax2.set_xlabel("Epoch", color="#aaa"); ax2.set_ylabel("Loss", color="#aaa")
    ax2.tick_params(colors="#aaa"); ax2.spines[:].set_color("#333")
    ax2.set_title("As loss goes down, the network is learning! 🎓",
                  color="white", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Auto-play ─────────────────────────────────────────────────────────────────
st.divider()
col_auto1, col_auto2 = st.columns([1, 3])
with col_auto1:
    auto = st.toggle("▶️ Auto-play steps")
with col_auto2:
    speed = st.slider("Speed (seconds between steps)", 0.5, 3.0, 1.5, 0.5)

if auto and step < len(STEP_DESCRIPTIONS) - 1:
    time.sleep(speed)
    st.session_state.step += 1
    if st.session_state.step == 6:
        for li in range(len(weights)):
            weights[li] -= lr * weight_delta[li]
        st.session_state.weights      = weights
        st.session_state.epoch       += 1
        st.session_state.loss_history.append(loss)
        st.session_state.activations  = forward_pass(inputs, weights)
    st.rerun()
elif auto and step == len(STEP_DESCRIPTIONS) - 1:
    time.sleep(speed)
    st.session_state.step = 0
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<center style='color:#555; font-size:0.85rem'>
Built with ❤️ for curious students in India &nbsp;·&nbsp;
<em>"Every expert was once a beginner."</em>
</center>
""", unsafe_allow_html=True)
