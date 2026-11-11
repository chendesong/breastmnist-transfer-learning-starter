# BreastMNIST Data Support

Three files for checking access to the official BreastMNIST 28x28 dataset.
Use any Python editor or IDE. No notebook or GPU is required.
Follow the final assignment brief for model tasks, assessment and permitted AI use.

## Run

Python 3.13 is used for local verification. From this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_data.py
```

If the dependencies are already installed in your selected Python environment:

```text
python check_data.py
```

The first run downloads `data/breastmnist.npz` from the official Zenodo record.
Later runs reuse the cached file and check its MD5 again; they need no download.
A successful run prints `PASS: all dataset checks completed` and saves
`data/train_preview.png`, containing eight original training images.
Checks or downloads that fail return a non-zero exit code. A checksum mismatch
does not replace the existing archive: inspect or move that file before retrying.

To use a different cache location:

```text
python check_data.py --data-dir "C:/my-data/breastmnist"
```

## What is checked

- The archive matches the publisher's MD5: `750601b1f35ba3300ea97c75c52ff8f6`.
- Its six arrays have the official shapes and `uint8` types.
- Training, validation and test contain 546, 78 and 156 images, respectively.
- Images remain 28x28 grayscale; labels remain `0 = malignant`, `1 = normal/benign`.
- Class counts and pixel ranges are printed. Only training images are previewed.

These checks verify the specified archive and its structure. They do not validate
individual clinical annotations, patient independence or a student's model.
The script does not re-split, resize, normalise, train or evaluate anything.
Implement your own model, preprocessing and experiments, and install the libraries
they need. Reserve the test set for final evaluation.

Sources: [official metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py),
[official dataset](https://zenodo.org/records/10519652/files/breastmnist.npz?download=1).
