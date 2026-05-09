# 🧠 Neural Network Explorer

An interactive, animated Streamlit app that teaches school students **how neural networks learn** — step by step. Designed for classroom use in rural India, with no prior machine learning knowledge required.

---

## 📸 What It Shows

The app walks through 7 guided steps, animating a live neural network:

| Step | Concept |
|------|---------|
| 🖼️ Step 1 | **Input** — an image is broken into numbers |
| ➡️ Step 2 | **Forward Pass** — signals flow through the network |
| ⚡ Step 3 | **Activation** — neurons decide whether to "fire" |
| 🏁 Step 4 | **Output** — the network makes a prediction |
| ❌ Step 5 | **Loss** — we measure how wrong it was |
| 🔙 Step 6 | **Backpropagation** — the error flows backwards |
| 🔧 Step 7 | **Weight Update** — the network adjusts and learns |

A **loss curve** builds up as students repeat the training steps, letting them literally watch the network get smarter.

---

## 🚀 Getting Started

### 1. Clone or download the files

Make sure you have these two files in the same folder:

```
neural_network_animation.py
requirements.txt
```

### 2. Install Python

You need **Python 3.9 or higher**. Check your version:

```bash
python --version
```

Download Python from [python.org](https://www.python.org/downloads/) if needed.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run neural_network_animation.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## 🖥️ System Requirements

| | Minimum |
|---|---|
| Python | 3.9+ |
| RAM | 512 MB |
| Internet | Only needed for Google Fonts (optional) |
| OS | Windows, macOS, or Linux |

Works on low-end laptops. No GPU required.

---

## 🎓 How to Use in the Classroom

### Projector Demo (Teacher-led)
1. Run the app on the teacher's laptop, project it on screen
2. Enable **Auto-play** from the bottom of the page
3. Set speed to **2–3 seconds** per step so students can follow
4. After all 7 steps, click **Next** again to repeat — the loss curve will grow

### Student Exploration
1. Each student (or pair) runs the app on their own device
2. Ask them to try all three input images: Cat 🐱, Dog 🐶, Car 🚗
3. Have them click through all 7 steps and watch how predictions change
4. Challenge: *"What happens to the loss when you click through many times?"*

### Discussion Questions
- Why does the loss go down over time?
- What happens if you set the Learning Rate very high (close to 1.0)?
- What does it mean when a connection is red vs blue?
- Can you make the network predict "Cat" more confidently?

---

## 🎛️ Controls

| Control | What it does |
|---------|-------------|
| **◀ Prev / Next ▶** | Step through the 7 phases manually |
| **Choose an input image** | Switch between Cat, Dog, Car inputs |
| **Learning Rate slider** | Controls how big each weight update is |
| **Auto-play toggle** | Automatically advances through steps |
| **Speed slider** | Sets seconds between auto-play steps |
| **🔄 Reset Network** | Resets all weights and the loss curve |

---

## 🗂️ File Structure

```
📁 your-folder/
├── neural_network_animation.py   # Main app
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔬 What's Under the Hood

The network has this architecture:

```
Input (3)  →  Hidden 1 (4)  →  Hidden 2 (4)  →  Output (2)
```

- **Activation function:** ReLU (Rectified Linear Unit)
- **Output:** Softmax probabilities
- **Loss function:** Mean Squared Error (MSE)
- **Optimiser:** Gradient Descent

All weights are initialised randomly and update in real time as students click through the backpropagation step.

---

## 🛠️ Troubleshooting

**`streamlit: command not found`**
```bash
python -m streamlit run neural_network_animation.py
```

**Port already in use**
```bash
streamlit run neural_network_animation.py --server.port 8502
```

**Fonts not loading (no internet)**
The app still works fully — it falls back to system fonts automatically.

**Slow on old hardware**
Reduce the number of auto-play steps by using Manual mode (◀ ▶ buttons).

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `matplotlib` | Network diagrams and loss curve |
| `numpy` | Matrix math for forward pass & backprop |
| `Pillow` | Image handling (used by Streamlit internally) |

---

## 📄 Licence

Free to use for educational purposes. Share freely with students and teachers. ❤️

---

*"Every expert was once a beginner."*
