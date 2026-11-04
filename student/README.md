# BreastMNIST transfer learning starter

This optional scaffold helps you implement the neural-model part of the assignment. Follow the final brief for assessed requirements.

**The dataset is fixed; the architecture is your choice.** Use BreastMNIST's official train/validation/test splits. Labels are 0 = malignant and 1 = normal/benign.

Select an architecture with available pretrained weights and document your architecture and weight source. Implement build_model and input transforms appropriate to those weights. To investigate pretraining, compare pretrained and randomly initialised versions of the **same chosen architecture**, using consistent inputs and trainable layers.

The helpers provide data loading and metrics. Complete the three TODOs: adapt the binary model, train one epoch, and return prediction scores. Choose and explain learning rates, augmentation and training budget. Defaults are starting points; develop using validation data and reserve testing for final evaluation.

You may explore different models. Explain your selection and report relevant trials. Follow the final brief's AI-use and acknowledgement rules.

## Setup

Python 3.13 and these CPU versions were used for staff checks. From this folder, run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
```

Select this environment in your notebook editor. If needed, optionally register its kernel:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ipykernel install --user --name breastmnist-starter --display-name "BreastMNIST starter"
```

Open BreastMNIST_Starter.ipynb, complete the TODOs and enable the experiment switch. Keep starter_utils.py beside it. First use downloads the official data and chosen model's weights into local caches. A short run verifies the workflow; it does not establish comparative performance.
