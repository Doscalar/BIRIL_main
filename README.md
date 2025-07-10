# Deep Spiking Neural Network with Brain-Inspired Recurrent Iterative Learning (BIRIL)

![BIRIL Architecture](https://github.com/Doscalar/BIRIL_main/blob/main/figure/BIRIL.jpg)

Official implementation of the paper **"Deep Spiking Neural Network with Brain-Inspired Recurrent Iterative Learning"** - a novel hybrid learning framework that synergistically integrates biologically realistic spike transmission with adaptive excitation-inhibition dynamics.

## 🔍 Overview

BIRIL is a brain-inspired training framework for Spiking Neural Networks (SNNs) that:
- Emulates biological excitation-inhibition balance (∼4:1 ratio)
- Combines STDP-based local learning with STBP-based global learning
- Dynamically adjusts neuronal activity through a three-cycle training strategy
- Achieves state-of-the-art performance with low computational overhead

```python
# Core training mechanism
def train_BIRIL():
    excitation_cycle = simulate_biological_excitation()
    inhibition_cycle = apply_neural_inhibition()
    hybrid_cycle = integrate_ei_balance()
    return optimize_network(electro-chemical_conversion)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- SpikingJelly framework

### Installation

```bash
git clone https://github.com/yourusername/BIRIL-SNN.git
pip install -r requirements.txt
```


## 📊 Key Results

### Comparative Performance (Accuracy %)

| Dataset        | STDP   | STDP-STBP | BIRIL  | Improvement vs SOTA |
|----------------|:------:|:---------:|:------:|:-------------------:|
| CIFAR-10       | 42.23  |   93.60   | 93.40  |       +4.46%        |
| CIFAR-100      | 11.54  |   72.25   | 70.89  |       +18.15%       |
| MNIST          | 89.16  |   99.26   | 99.30  |       +0.40%        |
| DVS128 Gesture | 24.31  |   94.80   | 95.49  |       +8.68%        |

### Optimal BIRIL Ratios

| Dataset        | Best Ratio (STDP:STBP:STDP-STBP) | Accuracy |
|----------------|:--------------------------------:|:--------:|
| CIFAR-10       |               1:8:3              |  93.40%  |
| CIFAR-100      |               1:8:3              |  70.89%  |
| MNIST          |               1:6:2              |  99.30%  |
| DVS128 Gesture |               2:6:1              |  95.49%  |





