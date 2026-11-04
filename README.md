# BreastMNIST Transfer Learning Starter

A draft teaching scaffold for medical image classification and transfer learning. The notebook provides a starting workflow while leaving core implementation and experimental decisions to students. Follow the final assignment brief for tasks, assessment and permitted AI assistance.

**The dataset is fixed; the model and weights are your choice.** BreastMNIST contains 28×28 grayscale breast ultrasound images for binary classification. Keep the official training, validation and test splits: label 0 is malignant; label 1 is normal/benign.

| Material | Purpose |
|---|---|
| [Student notebook](student/BreastMNIST_Starter.ipynb) | Experiment workflow and three TODOs |
| [Setup and running instructions](student/README.md) | Environment, dependencies and execution |
| [Verification notes](docs/verification.md) | Scope and limitations of local checks |
| [Downloadable starter package](student_starter.zip) | Files for local use |

## Student work

Select an architecture with available pretrained weights, document the model and weight source, and adapt its input processing. Compare pretrained and randomly initialised versions of the **same architecture**, keeping input processing and trainable layers consistent.

Complete three TODOs:

1. Adapt the binary classification model.
2. Implement one training epoch.
3. Return malignant-class prediction scores.

Choose and explain learning rates, augmentation and training choices. You may explore models; explain your selection and report relevant trials. Analyse validation results and errors using your own experiments.

## Getting started

Read the setup instructions, open the notebook, run environment and data checks, complete the TODOs, and enable experiments. First use downloads the dataset and chosen weights into local caches.

Use validation data during development and evaluate the fixed experiment plan on the final test split. Short runs check the workflow; they do not establish model superiority. Upsampling adds no missing image detail, and results do not establish clinical usefulness.

Sources: [MedMNIST](https://github.com/MedMNIST/MedMNIST) and the [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
