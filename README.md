# KouMed – Stroke Detection with Deep Learning (TEKNOFEST 2025)

KouMed is a deep learning project developed by the KOÜ-MED team for the TEKNOFEST 2025 Artificial Intelligence in Health Competition. This project uses state-of-the-art deep learning architectures to detect brain strokes from CT images.

## 🧠 Project Description

We aim to create an AI-based diagnostic tool capable of detecting strokes in brain CT scans with high accuracy and clinical reliability. The project covers the full AI development pipeline, including:

* Dataset preprocessing
* Deep learning model development
* Data augmentation
* Hyperparameter tuning
* Evaluation using multiple performance metrics

## 🚀 Highlights

* Successfully passed the **first stage** of TEKNOFEST with a score of **86 points**
* Utilized both **TensorFlow** and **PyTorch** frameworks
* Designed custom **CNN models**
* Implemented segmentation models like **U-Net** and **U-Net++**
* Explored transformer-based architectures:

  * **ViT** (Vision Transformer)
  * **DeiT** (Data-efficient Image Transformers)
  * **Swin Transformer**
  * **BEiT** (Bidirectional Encoder Representation from Image Transformers)
* Integrated **EfficientNet** for scalable image classification
* Applied advanced data augmentation and optimization techniques to boost generalization

## 🛠️ Technologies Used

* Python
* TensorFlow
* PyTorch
* NumPy, Pandas
* Scikit-learn
* Matplotlib

## 📊 Evaluation Metrics

* Accuracy
* F1-Score
* Precision
* Recall
* Confusion Matrix

## 📁 Implemented Models

The following deep learning models were implemented and tested:

* `CNN.ipynb`: Convolutional Neural Network custom main model
* `U-Net.ipynb`: U-Net for medical image segmentation
* `U-Net++.ipynb`: Enhanced version of U-Net with nested skip connections
* `EfficientNet.ipynb`: EfficientNet for optimized classification
* `Vit Transformer.ipynb`: Vision Transformer for stroke classification
* `DeiT Transformer.ipynb`: Data-efficient Transformer with knowledge distillation
* `Swin Transformer.ipynb`: Swin Transformer using hierarchical feature maps
* `BEiT Transformer.ipynb`: BEiT for self-supervised pretraining on medical images

## 👥 Team Collaboration

This project was developed in collaboration with a multidisciplinary team to ensure that both technical and clinical standards were met.

## 📌 Project Status

* ✅ First stage: Completed (Score: 86)
* 🚧 Second stage: In development

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*Developed with ❤️ by the KOÜ-MED team for TEKNOFEST 2025.*
