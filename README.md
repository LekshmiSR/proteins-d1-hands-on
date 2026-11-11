# Hands-on deep learning for protein secondary structure

Step-by-step, beginner-friendly notebooks that train four classic deep learning architectures on the **same protein classification problem**, so you can see exactly what goes in and out of every layer and compare the models fairly.

* **Task:** for every amino acid in a protein, predict its secondary structure: **H** (helix), **E** (strand) or **C** (coil). This is a per-residue, 3-class classification problem (called *Q3*).
* **Data:** the [PS4 dataset](https://github.com/omarperacha/ps4-dataset) (18,731 non-redundant protein chains from the PDB with DSSP labels) for training and validation, and the classic **CB513** benchmark as an independent test set. About 10 MB, downloaded automatically.
* **Models:** MLP → 1D CNN → bidirectional LSTM → Transformer, each in its own notebook.
* **Hardware:** runs on a free Google Colab **T4 GPU** (each notebook takes a few minutes). All models also fit a 4 GB laptop GPU such as a GTX 1650.

No installation and no code editor are needed: click a badge, and the notebook opens in your browser.

## The notebooks

Do them in order: each one builds on the previous one.

| # | Notebook | What you learn | Colab |
|---|---|---|---|
| 00 | [Data and task](notebooks/00_data_and_task.ipynb) | The biology, the datasets, tokens and embeddings, padding and masks, metrics, two baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/00_data_and_task.ipynb) |
| 01 | [MLP with a sliding window](notebooks/01_mlp_sliding_window.ipynb) | Fully connected layers, windows, **one training step by hand** (loss, gradients, optimizer) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/01_mlp_sliding_window.ipynb) |
| 02 | [1D CNN](notebooks/02_cnn_1d.ipynb) | Convolutions, channels, weight sharing, receptive field | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/02_cnn_1d.ipynb) |
| 03 | [Bidirectional LSTM](notebooks/03_bilstm.ipynb) | Recurrence, LSTM gates, bidirectionality, packed sequences | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/03_bilstm.ipynb) |
| 04 | [Transformer](notebooks/04_transformer.ipynb) | Positional encoding, self-attention by hand, multi-head attention, attention maps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/04_transformer.ipynb) |
| 05 | [Compare the models](notebooks/05_compare_models.ipynb) | Summary table, per-class scores, confusion matrices, take-home messages | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LekshmiSR/proteins-d1-hands-on/blob/main/notebooks/05_compare_models.ipynb) |

Every model notebook follows the same pattern:
1. **Settings**: all the knobs in one cell.
2. **Build the model step by step**: one batch goes through each layer and the tensor **shape is printed after every step**.
3. **The model class**: the same steps packed into a PyTorch `nn.Module`.
4. **Train**, then plot the learning curves.
5. **Evaluate** on CB513: Q3 accuracy, per-class precision/recall/F1, confusion matrix, an example prediction.
6. **Save** the results for notebook 05, and **exercises** to try.

## How to run in Google Colab (recommended)

1. Click an **Open in Colab** badge above. Sign in with a Google account if asked.
2. Turn on the free GPU: menu **Runtime → Change runtime type → T4 GPU → Save**.
3. Run the cells from top to bottom: click a cell and press **Shift + Enter**, or use **Runtime → Run all**.
4. If Colab warns "This notebook was not authored by Google", click **Run anyway**.

The first cell downloads this repository into Colab, so the helper code in `protein_dl/` is available. Nothing is installed on your computer.

**Keeping your results.** Colab deletes all files when the session ends. To compare models you trained on different days, set `SAVE_TO_DRIVE = True` in the settings cell of notebooks 01–04 (Colab will ask for permission to use your Google Drive), and `USE_DRIVE = True` in notebook 05. If you skip this, notebook 05 still works using the reference results shipped in `results/reference/`.

**Quick try.** Set `TRAIN_SUBSET = 2000` in the settings cell to train on 2,000 proteins only. It is much faster (even on a CPU) and great for experimenting, but the scores will be lower.

## Reference results

_Reference results are being computed and will be added here._

## How to run on your own computer (optional)

You can use Jupyter Notebook from Anaconda; no VS Code needed.

1. Download this repository: green **Code** button → **Download ZIP**, then unzip it (or `git clone` it).
2. Open **Anaconda Prompt** and create an environment:
   ```
   conda create -n protein-dl python=3.11 -y
   conda activate protein-dl
   ```
3. Install PyTorch **with GPU support**. Go to <https://pytorch.org/get-started/locally/>, choose *Stable, Windows, Pip, Python, CUDA 12.x*, and run the command it shows (it looks like `pip install torch --index-url https://download.pytorch.org/whl/cu12x`).
4. Install the rest and start Jupyter:
   ```
   cd path\to\the\unzipped\folder
   pip install numpy pandas scikit-learn matplotlib jupyter
   jupyter notebook
   ```
5. In the browser window that opens, go into `notebooks/` and open `00_data_and_task.ipynb`.

Check that the GPU is found: the first cells of notebooks 01–04 print `Using GPU: NVIDIA GeForce GTX 1650`. If you run out of GPU memory, lower `BATCH_SIZE` (e.g. 16) or `MAX_LEN` (e.g. 256) in the settings cell.

## What is in the repository

```
notebooks/          the six notebooks (start here)
protein_dl/
  data.py           download, encoding, Dataset, padding (explained in notebook 00)
  train.py          training loop, evaluation, metrics, saving results (explained in notebook 01)
  plots.py          learning curves, confusion matrix, prediction printout
results/reference/  results of the four models with default settings
```

## The dataset in more detail

| Split | Source | Proteins | Scored residues |
|---|---|---|---|
| train | PS4 official training split | 17,799 | ~3.8 million |
| validation | PS4 official validation split | 932 | ~0.2 million |
| test | CB513 (NetSurfP-2.0 version) | 511 | 83,991 |

DSSP 8-state labels are reduced to 3 states in the standard way: H, G, I → **H**; E, B → **E**; everything else (T, S, C, P) → **C**. The data files are downloaded from a fixed commit of the PS4 repository, so everyone gets exactly the same data.

## References

* Peracha O. *PS4: a next-generation dataset for protein single-sequence secondary structure prediction.* BioTechniques (2023). Data: <https://github.com/omarperacha/ps4-dataset> (CC0).
* Cuff J.A., Barton G.J. *Evaluation and improvement of multiple sequence methods for protein secondary structure prediction.* Proteins 34:508–519 (1999). (CB513)
* Klausen M.S. et al. *NetSurfP-2.0: improved prediction of protein structural features by integrated deep learning.* Proteins 87:520–527 (2019). (CB513 file used here)
* Kabsch W., Sander C. *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features.* Biopolymers 22:2577–2637 (1983). (DSSP)
* Vaswani A. et al. *Attention is all you need.* NeurIPS (2017). (Transformer)
* Hochreiter S., Schmidhuber J. *Long short-term memory.* Neural Computation 9:1735–1780 (1997). (LSTM)

## Licence

Code: MIT (see `LICENSE`). Data: PS4 is released under CC0 by its authors; it is downloaded from the original repository, not redistributed here.
