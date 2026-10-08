| Model | Test acc | Test macro-F1 | Val macro-F1 | Best epoch | Train (min) |
|---|---|---|---|---|---|
| HOG + PCA + Decision Tree (course baseline) | 33.3 | 31.6 | 29.4 | – | 2.3 |
| HOG + PCA + RBF-SVM | 67.7 | 66.9 | 62.3 | – | 1.5 |
| AlexNet, course recipe | 23.5 | 6.4 | 6.3 | 23 / 30 | 2.0 |
| AlexNet from scratch, improved recipe | 83.6 | 82.3 | 86.6 | 99 / 100 | 16.1 |
| AlexNet-BN from scratch, improved recipe | 79.9 | 77.1 | 82.9 | 73 / 98 | 15.5 |
| AlexNet ImageNet-pretrained, fine-tuned | 88.4 | 86.6 | 91.2 | 30 / 30 | 4.8 |

Per-class test F1 (%):

| Model | cardboard | glass | metal | paper | plastic | trash |
|---|---|---|---|---|---|---|
| HOG + PCA + Decision Tree (course baseline) | 39.3 | 36.6 | 29.2 | 41.1 | 24.0 | 19.0 |
| HOG + PCA + RBF-SVM | 72.3 | 58.8 | 62.9 | 79.8 | 62.9 | 64.7 |
| AlexNet, course recipe | 0.0 | 0.0 | 0.0 | 38.1 | 0.0 | 0.0 |
| AlexNet from scratch, improved recipe | 90.3 | 77.0 | 80.3 | 92.8 | 79.5 | 73.9 |
| AlexNet-BN from scratch, improved recipe | 88.7 | 73.4 | 81.2 | 89.7 | 75.2 | 54.2 |
| AlexNet ImageNet-pretrained, fine-tuned | 92.2 | 85.0 | 90.0 | 91.8 | 87.9 | 72.7 |
