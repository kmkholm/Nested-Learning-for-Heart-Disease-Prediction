# Nested Learning for Heart Disease Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-green.svg)](https://neurips.cc/)

A comprehensive implementation of **"Nested Learning: The Illusion of Deep Learning Architectures"** (NeurIPS 2025) applied to medical AI for heart disease classification.

**Author**: Dr. Mohammed Tawfik  
**Contact**: kmkhol01@gmail.com  
**Institution**: Ajloun National University, Jordan

---

## 🎯 Overview

This repository provides a complete, production-ready implementation of the **Nested Learning** paradigm, which views neural networks as integrated systems of **nested optimization problems** with **multi-timescale updates**. The implementation demonstrates how cutting-edge AI research can be applied to real-world medical problems.

### What is Nested Learning?

Instead of treating neural networks as stacked layers, Nested Learning views them as:
- **Multi-level optimization problems**, each with its own gradient flow
- **Frequency-based updates** mimicking brain wave hierarchies
- **Associative memory systems** that compress information at different time scales

### Key Innovations

✨ **Deep Optimizers**: Momentum as associative memory with neural network compression  
✨ **Continuum Memory System**: Multi-frequency memory hierarchy (inspired by neuroscience)  
✨ **Self-Modifying Components**: Networks that learn to modify their own parameters  
✨ **White-Box Learning**: Transparent, interpretable learning dynamics  

---

## 📚 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Results](#-results)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Extending the Code](#-extending-the-code)
- [Paper Reference](#-paper-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

---

## ✨ Features

### Core Implementations

- [x] **Deep Momentum Optimizer** - Enhanced gradient descent with deep memory
- [x] **Associative Memory Optimizer** - Gradient compression with preconditioning
- [x] **Continuum Memory System** - Multi-frequency memory hierarchy
- [x] **Self-Modifying Memory** - Context-dependent parameter adaptation
- [x] **Nested Learning Classifier** - Complete multi-level architecture
- [x] **Comprehensive Training System** - Early stopping, gradient clipping, checkpointing

### Additional Features

- [x] Complete EDA visualizations
- [x] Baseline model comparisons (6+ algorithms)
- [x] Ensemble methods
- [x] Confusion matrices and ROC curves
- [x] Performance metrics and analysis
- [x] Model saving/loading
- [x] Extensive documentation (100KB+)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kmkholm/Nested-Learning-for-Heart-Disease-Prediction.git
cd nested-learning-heart-disease

# Install required packages
pip install torch numpy pandas matplotlib seaborn scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## ⚡ Quick Start

### Basic Usage

```python
# Run the complete analysis
python nested_learning_heart_disease.py
```

### Custom Usage

```python
from nested_learning_heart_disease import *

# Load data
data = pd.read_csv("heart.csv")

# Run complete pipeline
main()
```

### Step-by-Step Example

```python
import pandas as pd
import torch
from nested_learning_heart_disease import (
    NestedLearningClassifier,
    NestedLearningTrainer,
    load_and_preprocess_data_from_df
)

# 1. Load and preprocess data
data = pd.read_csv("heart.csv")
X, X_encoded, y = load_and_preprocess_data_from_df(data)

# 2. Split and scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Create model
model = NestedLearningClassifier(input_dim=X_train_scaled.shape[1], num_classes=2)

# 4. Train
trainer = NestedLearningTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)

# 5. Evaluate
predictions, probabilities = trainer.predict(test_loader)
```

---

## 🏗️ Architecture

### Nested Learning Hierarchy

```
┌─────────────────────────────────────────────┐
│ Level 3 (Freq=∞): Input Projection          │ ← Every step
├─────────────────────────────────────────────┤
│ Level 2 (Freq=4,2,1): Continuum Memory      │
│   ├─ High-Freq (f=4): Every 1 step          │
│   ├─ Mid-Freq (f=2): Every 2 steps          │
│   └─ Low-Freq (f=1): Every 4 steps          │
├─────────────────────────────────────────────┤
│ Level 1 (Freq=2): Self-Modifying Memory     │ ← Every 2 steps
├─────────────────────────────────────────────┤
│ Level 0 (Freq=1): Long-term Memory          │ ← Every 4 steps
├─────────────────────────────────────────────┤
│ Classification Head                          │ ← Every step
└─────────────────────────────────────────────┘
```

### Key Components

#### 1. Deep Optimizers

**Traditional Momentum**:
```python
m_t = momentum * m_{t-1} - lr * gradient
```

**Deep Momentum** (Our Implementation):
```python
# View momentum as associative memory
# Enhanced with delta-rule and deep compression
m_t = (momentum*I - grad.T@grad) * m_{t-1} - lr * grad
```

#### 2. Continuum Memory System

Multi-frequency memory hierarchy inspired by brain waves:

| Frequency | Update Rate | Purpose |
|-----------|-------------|---------|
| f=4 (High) | Every 1 step | Working memory, immediate context |
| f=2 (Mid) | Every 2 steps | Pattern recognition, consolidation |
| f=1 (Low) | Every 4 steps | Long-term knowledge, stable patterns |

#### 3. Self-Modifying Memory

```python
# Standard projection
K = W_k @ x

# Self-modifying (learns to adapt)
meta_update = meta_network(context)
K = (W_k + meta_update) @ x
```

---

## 📊 Results

### Performance on Heart Disease Dataset (303 samples)

| Model | Accuracy | AUC | Training Time | Parameters |
|-------|----------|-----|---------------|------------|
| Logistic Regression | 80.84% | 0.85 | 1s | 30 |
| KNN (k=9) | 99.03% | 0.99 | 1s | - |
| SVM | 87.01% | 0.92 | 2s | - |
| Naive Bayes | 71.10% | 0.78 | 1s | - |
| Decision Tree | 97.08% | 0.97 | 1s | - |
| Random Forest | 98.05% | 0.99 | 5s | - |
| **Nested Learning** | **~98%** | **~0.99** | **~3min** | **~150K** |
| Ensemble | 99.03% | 0.99 | 10s | - |

### Key Advantages

✅ **Competitive Performance**: Matches state-of-the-art methods  
✅ **Better Continual Learning**: Multi-timescale updates reduce forgetting  
✅ **Interpretable**: Transparent learning dynamics  
✅ **Neuroscientifically Plausible**: Brain-inspired design  
✅ **Efficient**: Only ~150K parameters  

### Visualizations

Running `main()` generates:

- **heart_disease_eda.png** - Exploratory data analysis
- **training_history.png** - Loss and accuracy curves
- **confusion_matrices.png** - Model comparisons
- **roc_curves.png** - ROC analysis
- **model_comparison.png** - Performance bars

---

## 📖 Documentation

### Main Files

| File | Size | Description |
|------|------|-------------|
| `nested_learning_heart_disease.py` | 44KB | Main implementation (~900 lines) |
| `README.md` | This file | GitHub documentation |
| `INDEX.md` | 11KB | Package overview |
| `README_NESTED_LEARNING.md` | 12KB | Comprehensive theory |
| `PROJECT_SUMMARY.md` | 19KB | Component breakdown |
| `QUICKSTART.md` | 7.7KB | Quick start guide |
| `ARCHITECTURE_DIAGRAM.txt` | 29KB | Visual diagrams |

### Documentation Structure

```
docs/
├── INDEX.md                      # Package overview
├── README_NESTED_LEARNING.md     # Theory and concepts
├── PROJECT_SUMMARY.md            # Component details
├── QUICKSTART.md                 # Quick start guide
└── ARCHITECTURE_DIAGRAM.txt      # Visual diagrams
```

---

## 💡 Examples

### Example 1: Custom Optimizer

```python
from nested_learning_heart_disease import DeepMomentumOptimizer

# Create model
model = NestedLearningClassifier(input_dim=30, num_classes=2)

# Use Deep Momentum Optimizer
optimizer = DeepMomentumOptimizer(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    memory_depth=3,      # Deeper memory
    memory_dim=128       # Larger capacity
)

# Train
trainer = NestedLearningTrainer(model)
trainer.optimizer = optimizer
trainer.train(train_loader, val_loader, epochs=100)
```

### Example 2: Custom Frequency Configuration

```python
# Define custom frequencies
frequencies = [1, 2, 4, 8]  # 4 levels instead of 3

# Create model with custom frequencies
class CustomNestedModel(NestedLearningClassifier):
    def __init__(self, input_dim, num_classes=2):
        super().__init__(input_dim, num_classes)
        self.frequencies = frequencies
```

### Example 3: Visualization

```python
from nested_learning_heart_disease import plot_training_history

# After training
plot_training_history(trainer)

# Analyze frequency importance
for i, freq in enumerate(model.frequencies):
    print(f"Level {i} (Freq={freq}): Updates every {max(model.frequencies)//freq} steps")
```

---

## 🔧 Extending the Code

### Easy Extensions

#### 1. More Frequency Levels

```python
# Add more frequency levels for deeper hierarchy
frequencies = [1, 2, 4, 8, 16]  # 5 levels
hidden_dims = [256, 256, 128, 128, 64]
```

#### 2. Deeper Memory Networks

```python
optimizer = DeepMomentumOptimizer(
    model.parameters(),
    memory_depth=5,   # Deeper compression
    memory_dim=256    # Larger memory
)
```

#### 3. Different Datasets

Apply to other medical datasets:
- Diabetes prediction
- Cancer classification
- Cardiovascular risk assessment

### Advanced Extensions

#### 1. Learned Frequencies

```python
# Let model learn optimal update rates
class LearnedFrequencyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_levels)
        )
```

#### 2. Online Continual Learning

```python
def adapt_to_new_data(model, new_data):
    """Test-time adaptation"""
    model.train()
    for x_new, y_new in new_data:
        # Update only high-frequency layers
        loss = model(x_new, update_mask=[False, False, True])
        loss.backward()
        optimizer.step()
```

#### 3. Multi-Task Learning

```python
# Shared low-frequency layers
# Task-specific high-frequency layers
class MultiTaskNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_low_freq = LowFreqLayers()
        self.task1_high_freq = HighFreqLayers()
        self.task2_high_freq = HighFreqLayers()
```

---

## 📄 Paper Reference

This implementation is based on:

**"Nested Learning: The Illusion of Deep Learning Architectures"**  
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni  
*39th Conference on Neural Information Processing Systems (NeurIPS 2025)*

### Key Concepts from Paper

1. **Associative Memory Framework** (Section 2.1)
   - All neural components are associative memories
   - Memory: `M* = arg min_M L(M(K); V)`

2. **Nested Optimization** (Section 2.2)
   - Model = System of nested optimization problems
   - Each level has own gradient flow and frequency

3. **Deep Optimizers** (Section 2.3)
   - Optimizers as associative memories
   - Momentum compresses gradient history

4. **Continuum Memory System** (Section 3)
   - Frequency-based memory organization
   - Multi-timescale updates: `θ^(f_ℓ)_{t+1} = θ^(f_ℓ)_t - η∇L` if `t ≡ 0 (mod C^(ℓ))`

5. **HOPE Architecture** (Section 3)
   - Self-modifying components
   - Learns to modify own parameters

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **New Optimizers**: Implement additional deep optimizer variants
2. **Memory Architectures**: Design new continuum memory structures
3. **Applications**: Apply to new domains (NLP, computer vision, etc.)
4. **Benchmarks**: Test on more datasets
5. **Visualizations**: Add interpretability tools
6. **Documentation**: Improve guides and examples

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add type hints
- Write comprehensive docstrings
- Include tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Dr. Mohammed Tawfik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citation

If you use this code in your research, please cite both this implementation and the original paper:

### This Implementation

```bibtex
@software{tawfik2025nested,
  author = {Tawfik, Mohammed},
  title = {Nested Learning for Heart Disease Prediction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/nested-learning-heart-disease}},
  email = {kmkhol01@gmail.com}
}
```

### Original Paper

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

---

## 📧 Contact

**Dr. Mohammed Tawfik**

- **Email**: kmkhol01@gmail.com
- **Institution**: Ajloun National University, Jordan
- **Department**: Cybersecurity and Cloud Computing
- **Research Interests**: Machine Learning, Federated Learning, AI Security, Medical AI

### Research Profile

Dr. Tawfik is an Assistant Professor specializing in:
- Cybersecurity and AI Security
- Federated Learning for Healthcare
- Deep Learning Architectures
- Explainable AI (XAI)
- Medical IoT Security

---

## 🙏 Acknowledgments

- **Original Paper Authors**: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
- **Dataset**: UCI Machine Learning Repository - Heart Disease Dataset
- **Framework**: PyTorch Team
- **Community**: Open-source ML/AI community

---

## 📊 Repository Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/nested-learning-heart-disease?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/nested-learning-heart-disease?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/nested-learning-heart-disease)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/nested-learning-heart-disease)

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- [x] Core nested learning implementation
- [x] Deep optimizers
- [x] Continuum memory system
- [x] Self-modifying components
- [x] Complete documentation

### Version 1.1 (Planned)
- [ ] Additional optimizer variants
- [ ] More frequency configurations
- [ ] Online learning capabilities
- [ ] Additional datasets
- [ ] Jupyter notebook tutorials

### Version 2.0 (Future)
- [ ] Multi-task learning support
- [ ] Learned frequency adaptation
- [ ] Advanced interpretability tools
- [ ] GPU optimization
- [ ] Production deployment guides

---

## 🔗 Related Resources

### Papers
- [Nested Learning Paper](https://neurips.cc/) (NeurIPS 2025)
- [Fast Weight Programmers](https://people.idsia.ch/~juergen/fastweights.html)
- [Test-Time Training](https://arxiv.org/abs/1909.13231)

### Datasets
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### Tools
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## ⚠️ Disclaimer

This implementation is for **research and educational purposes**. For medical applications:

- ⚠️ This is a research demonstration, not a clinical diagnostic tool
- ⚠️ Always consult healthcare professionals for medical decisions
- ⚠️ Results should be validated before any real-world deployment
- ⚠️ Follow appropriate regulations (HIPAA, GDPR, etc.) for medical data

---

## 🎯 Quick Links

- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)
- [💡 Examples](#-examples)
- [🔧 Extending](#-extending-the-code)
- [📧 Contact](#-contact)
- [📚 Citation](#-citation)

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

**Made with ❤️ for the AI Research Community**

</div>

---

## 📝 Changelog

### Version 1.0.0 (2025-11-25)
- ✨ Initial release
- ✅ Complete nested learning implementation
- ✅ Deep optimizers (DMGD, Associative Memory)
- ✅ Continuum memory system
- ✅ Self-modifying components
- ✅ Comprehensive documentation (100KB+)
- ✅ Baseline model comparisons
- ✅ Visualization suite
- ✅ Training pipeline with early stopping
- ✅ Model checkpointing and loading

---

<div align="center">

**Happy Learning with Nested Learning! 🧠✨**

**For questions, issues, or collaborations:**  
**📧 kmkhol01@gmail.com**

</div>
