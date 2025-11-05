# 🧠 CIFAR-10 Image Classification using PyTorch

This project is my first deep learning experiment using **PyTorch**.  
I trained a **ResNet18** Convolutional Neural Network (CNN) on the **CIFAR-10 dataset** to classify images into 10 categories such as airplane, car, cat, and dog.

---

## 🎯 Objective
To build and train a neural network that can accurately classify small colored images from the CIFAR-10 dataset using GPU acceleration on Google Colab.

---

## ⚙️ Technologies Used
- **Python 3**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
- **Google Colab (GPU Runtime)**

---

## 🧩 Dataset
- **CIFAR-10**: 60,000 color images (32x32 pixels), 10 classes.
- Provided directly from `torchvision.datasets`.

---

## 🏗️ Model Architecture
- **ResNet18** — a convolutional neural network with residual blocks.
- Final fully connected layer modified for 10 output classes.
- Optimizer: Adam  
- Loss function: CrossEntropyLoss  
- Training epochs: 5  

---

## 📊 Results
- **Training Accuracy:** ~85%  
- **Test Accuracy:** ~80–85%  
- Model file saved as `cifar10_resnet18.pth`.

Example output:

---

## 📷 Predictions (Sample)
Example of the model’s predictions vs actual labels:

| Image | Ground Truth | Predicted |
|-------|---------------|-----------|
| 🛩️ | airplane | airplane |
| 🚗 | car | car |
| 🐶 | dog | dog |

---

## 💾 How to Run
1. Open in [Google Colab](https://colab.research.google.com/).
2. Copy the code from the notebook or this repo.
3. Enable GPU under Runtime → Change runtime type → GPU.
4. Run all cells.
5. Accuracy and prediction results will appear at the end.

---

## 🧠 What I Learned
- Basics of deep learning and convolutional neural networks.
- How to use **PyTorch** for model training and evaluation.
- Importance of GPU acceleration for faster computation.
- How to publish and document projects on GitHub.

---

## 📈 Future Improvements
- Try pretrained models (ResNet50, EfficientNet).
- Add data augmentation to improve accuracy.
- Experiment with learning rate schedules and mixed precision training.

---

## 👤 Author
**Bunny**  
📍 First-year student exploring AI & Deep Learning.  
💬 “Start small. Learn fast. Build something cool!”
