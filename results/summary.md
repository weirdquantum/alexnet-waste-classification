| Model | Seeds | Test acc | Test macro-F1 | Val macro-F1 | Best epoch | Train (min) |
|---|---|---|---|---|---|---|
| HOG + PCA + Decision Tree (course baseline) | 1 | 34.7 | 31.6 | 31.3 | – | 1.0 |
| HOG + PCA + RBF-SVM | 1 | 63.7 | 62.1 | 62.6 | – | 0.7 |
| AlexNet, course recipe | 3 | 24.2 ± 0.0 | 6.5 ± 0.0 | 6.5 ± 0.0 | 13/30, 2/30, 22/30 | 0.3 |
| AlexNet from scratch, improved recipe | 3 | 80.4 ± 0.5 | 79.3 ± 0.2 | 83.8 ± 0.4 | 73/98, 67/92, 80/100 | 4.5 |
| AlexNet-BN from scratch, improved recipe | 3 | 79.1 ± 0.6 | 76.0 ± 1.2 | 80.3 ± 1.0 | 91/100, 87/100, 98/100 | 4.8 |
| AlexNet ImageNet-pretrained, fine-tuned | 3 | 86.7 ± 0.6 | 85.6 ± 0.2 | 88.3 ± 0.2 | 19/29, 23/30, 24/30 | 1.4 |

Per-class test F1 (%, mean over seeds):

| Model | cardboard | glass | metal | paper | plastic | trash |
|---|---|---|---|---|---|---|
| HOG + PCA + Decision Tree (course baseline) | 43.5 | 37.6 | 24.8 | 41.4 | 29.2 | 13.3 |
| HOG + PCA + RBF-SVM | 71.3 | 58.9 | 59.5 | 73.4 | 55.2 | 54.5 |
| AlexNet, course recipe | 0.0 | 0.0 | 0.0 | 39.0 | 0.0 | 0.0 |
| AlexNet from scratch, improved recipe | 86.6 | 75.3 | 77.3 | 89.3 | 73.3 | 74.2 |
| AlexNet-BN from scratch, improved recipe | 88.6 | 72.9 | 75.5 | 91.7 | 70.4 | 57.1 |
| AlexNet ImageNet-pretrained, fine-tuned | 95.3 | 80.7 | 80.5 | 94.2 | 83.4 | 79.6 |
