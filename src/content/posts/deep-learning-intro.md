---
title: "Deep Learning: Từ cơ bản đến ứng dụng thực tế"
published: 2025-06-29
description: "Khám phá deep learning từ neural networks cơ bản đến các ứng dụng AI hiện đại"
tags: ["Deep Learning"]
category: "AI & Machine Learning"
draft: false
lang: "vi"
---

## Mục lục

1. [Neural Networks cơ bản](#1-neural-networks-cơ-bản)
    - [Perceptron (Neuron đơn giản)](#perceptron-neuron-đơn-giản)
    - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
2. [Convolutional Neural Networks (CNN)](#2-convolutional-neural-networks-cnn)
    - [CNN cho Computer Vision](#cnn-cho-computer-vision)
    - [Transfer Learning với Pre-trained Models](#transfer-learning-với-pre-trained-models)
3. [Recurrent Neural Networks (RNN)](#3-recurrent-neural-networks-rnn)
    - [LSTM cho Text Classification](#lstm-cho-text-classification)
4. [Generative Adversarial Networks (GAN)](#4-generative-adversarial-networks-gan)
    - [Simple GAN](#simple-gan)
5. [Autoencoders](#5-autoencoders)
    - [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
6. [Attention Mechanisms](#6-attention-mechanisms)
    - [Self-Attention Implementation](#self-attention-implementation)
7. [Ứng dụng thực tế](#7-ứng-dụng-thực-tế)
    - [Image Classification với ResNet](#image-classification-với-resnet)
    - [Text Generation với GPT-style](#text-generation-với-gpt-style)
8. [Kết luận](#kết-luận)

# Deep Learning: Từ cơ bản đến ứng dụng thực tế 🧠

Deep Learning là một nhánh của Machine Learning sử dụng các neural networks với nhiều lớp để học các biểu diễn phức tạp từ dữ liệu. Trong bài viết này, tôi sẽ giới thiệu từ những khái niệm cơ bản đến các ứng dụng thực tế.

## 1. Neural Networks cơ bản

### Perceptron (Neuron đơn giản) với PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dữ liệu AND gate
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32)

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = Perceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print("Predictions:")
with torch.no_grad():
    preds = model(X)
    print(torch.round(preds).squeeze().numpy())
```

### Multi-Layer Perceptron (MLP) với PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Dữ liệu mẫu
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    test_outputs = model(X_test)
    acc = ((test_outputs > 0.5) == y_test).float().mean()
    print(f"Test Accuracy: {acc:.3f}")
```

## 2. Convolutional Neural Networks (CNN)

### CNN cho Computer Vision

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Chuẩn bị dữ liệu MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện
for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Đánh giá
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {correct/total:.3f}")
```

### Transfer Learning với Pre-trained Models (VGG16)

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

# Freeze base model
for param in vgg16.parameters():
    param.requires_grad = False

# Thay đổi classifier nếu muốn fine-tune
vgg16.classifier[6] = torch.nn.Linear(4096, 10)  # Ví dụ: 10 classes

# Dự đoán ảnh
def predict_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = vgg16(img)
        _, pred = torch.max(output, 1)
    return pred.item()
```

## 3. Recurrent Neural Networks (RNN)

### LSTM cho Text Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# Giả lập dữ liệu
texts = [
    "Tôi thích machine learning",
    "Deep learning rất thú vị",
    "AI là tương lai của công nghệ",
    "Thuật toán rất quan trọng",
    "Neural networks rất phức tạp"
]
labels = [1, 1, 1, 0, 0]

# Tokenization đơn giản
vocab = {}
def tokenize(text):
    return [vocab.setdefault(word, len(vocab)) for word in text.lower().split()]

sequences = [torch.tensor(tokenize(t)) for t in texts]
padded = pad_sequence(sequences, batch_first=True)
labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

model = LSTMClassifier(len(vocab))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(padded)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    preds = model(padded)
    print("Predictions:", (preds > 0.5).int().squeeze().tolist())
```

## 4. Generative Adversarial Networks (GAN)

### Simple GAN

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784), # 28x28 = 784
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Tạo models
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

# Compile
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# GAN
discriminator.trainable = False
gan_input = torch.randn(1, latent_dim)
gan_output = discriminator(generator(gan_input))
gan = torch.nn.Model(gan_input, gan_output)
gan.compile(optimizer=optimizer_g, loss=criterion)

print("GAN Architecture:")
print(gan)
```

## 5. Autoencoders

### Variational Autoencoder (VAE)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        x = self.model(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

# Sampling layer
def sampling(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784), # 28x28 = 784
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

# Build VAE
latent_dim = 2
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)

# VAE model
inputs = torch.randn(1, 784) # Example input
mu, log_var = encoder(inputs)
z = sampling(mu, log_var)
outputs = decoder(z)

vae = nn.Model(inputs, outputs)

# Loss function
reconstruction_loss = nn.MSELoss()(inputs, outputs)
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer=optimizer_g)

print("VAE Architecture:")
print(vae)
```

## 6. Attention Mechanisms

### Self-Attention Implementation

```python
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    # Tính attention scores
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale
    dk = query.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
    
    # Apply mask nếu có
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    
    # Apply attention weights
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Multi-head attention
def multi_head_attention(d_model, num_heads):
    def attention(inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Linear transformations
        query = torch.nn.Linear(d_model)(inputs)
        key = torch.nn.Linear(d_model)(inputs)
        value = torch.nn.Linear(d_model)(inputs)
        
        # Reshape for multi-head
        query = query.view(batch_size, seq_len, num_heads, d_model // num_heads)
        key = key.view(batch_size, seq_len, num_heads, d_model // num_heads)
        value = value.view(batch_size, seq_len, num_heads, d_model // num_heads)
        
        # Transpose for attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(
            query, key, value)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        return attention_output, attention_weights
    
    return attention

# Test
d_model = 512
num_heads = 8
attention_layer = multi_head_attention(d_model, num_heads) # có thể sử dụng nn.MultiheadAttention module

# Input shape: (batch_size, seq_len, d_model)
test_input = torch.randn(1, 10, d_model)
output, weights = attention_layer(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## 7. Ứng dụng thực tế

### Image Classification với ResNet

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Hàm dự đoán ảnh
def predict_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return pred.item()

# Ví dụ sử dụng (cần có ảnh thực tế)
# results = predict_image('path/to/image.jpg')
# print(f"Predicted class: {results}")
```

### Text Generation với GPT-style

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# tải pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=100):
    # Encode input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    outputs = model.generate(inputs, 
                           max_length=max_length,
                           num_return_sequences=1,
                           temperature=0.7,
                           do_sample=True)
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test
prompt = "Machine learning is"
generated = generate_text(prompt)
print(generated)
```

## Kết luận

Deep Learning đã mở ra những khả năng mới trong AI. Để thành công trong lĩnh vực này:

### 1. **Nền tảng vững chắc**
- Toán học: Linear Algebra, Calculus, Statistics
- Lập trình: Python, TensorFlow/PyTorch
- Kiến thức ML cơ bản


### 2. **Tools và Frameworks**
- **TensorFlow/PyTorch**: Deep learning
- **Transformers**: NLP
- **OpenCV**: Computer vision
- **JAX**: Research

