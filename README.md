# 7Gen - Advanced MNIST Digit Generation System

**State-of-the-art Conditional GAN for MNIST digit synthesis with self-attention mechanisms.**

---

## 🚀 Features

- 🎯 **Conditional Generation**: Generate specific digits (0–9) on demand.
- 🖼️ **High Quality Output**: Sharp and realistic handwritten digit samples.
- ⚡ **Fast Inference**: Real-time generation on GPU.
- 🔌 **Easy Integration**: Minimal setup, PyTorch-native implementation.
- 🚀 **GPU Acceleration**: Full CUDA support.

---

## 🔍 Model Details

- **Architecture**: Conditional GAN with self-attention  
- **Parameters**: 2.5M  
- **Input**: 100-dimensional noise vector + class label  
- **Output**: 28x28 grayscale images  
- **Training Data**: MNIST dataset (60,000 images)  
- **Training Time**: ~2 hours on NVIDIA RTX 3050 Ti  

---

## 🧪 Performance Metrics

| Metric           | Score |
|------------------|-------|
| **FID Score**    | 12.3  |
| **Inception Score** | 8.7   |

- **Training Epochs**: 100  
- **Batch Size**: 64  

---

## ⚙️ Training Configuration

```yaml
model:
  latent_dim: 100
  num_classes: 10
  generator_layers: [256, 512, 1024]
  discriminator_layers: [512, 256]

training:
  batch_size: 64
  learning_rate: 0.0002
  epochs: 100
  optimizer: Adam
  beta1: 0.5
  beta2: 0.999
