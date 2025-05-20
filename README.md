# Insider Threat Detection System

A graph-based and deep learning pipeline for detecting insider threats from enterprise user activity data.

> 🔗 **GitHub Link:** [https://github.com/Akobabs/InsiderThreatDetection](https://github.com/Akobabs/InsiderThreatDetection)

---

## 🚀 Overview

This project implements a comprehensive insider threat detection system using:

* **Graph Construction** via NetworkX
* **Node Embedding** using DeepWalk or Word2Vec
* **Session Grouping** for sequential modeling
* **LSTM Classification** via PyTorch

It processes user activity logs (logon, file access, HTTP requests, device connections) from the **CERT Insider Threat dataset (r5.2)** to classify potential threats.

---

## 🔧 Features

* **Graph Construction**: Multi-graph built from user activities across hosts, dates, and activity types.
* **HTTP Preprocessing**: Filters out non-sensitive and blacklisted domains.
* **Session Analysis**: Activity grouping with labeling for supervised training.
* **Embedding Generation**: Node2Vec-style embeddings from random walks.
* **Threat Detection**: LSTM model to classify suspicious user behavior.

---

## 📦 Requirements

### Software

* **Python**: 3.8 or higher
* **Dependencies**:
  Install via pip:

  ```bash
  pip install -r requirements.txt
  ```

**requirements.txt**

```
networkx>=2.5
numpy>=1.19
pandas>=1.1
torch>=1.8
scikit-learn>=0.24
matplotlib>=3.3
tqdm>=4.59
joblib>=1.0
```

### Hardware

* **GPU (Recommended)**: CUDA-compatible for model training
* **RAM**: ≥16GB for large graphs

---

## 📁 Dataset Setup

Place the CERT Insider Threat r5.2 dataset at:

```
/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/r5.2/
```

Expected files:

```
├── logon.csv
├── file.csv
├── http.csv
├── device.csv
├── answers/
│   ├── r5.2-1.csv
│   ├── r5.2-2.csv
│   └── ...
```

If your dataset path is different, **update `data_dir` and `data_version`** in the scripts.

---

## 🔍 Project Structure

```
insider-threat-detection/
│
├── preprocess_http_data.py     # Filters HTTP logs
├── graph_construct.py          # Builds activity graph
├── graph_embedding.py          # Creates embeddings
├── generate_data.py            # Creates session sequences
├── threat_detection.py         # Trains LSTM threat model
│
├── sequential_model.py         # LSTM model definition
├── walker.py                   # RandomWalk generators
├── classify.py                 # Embedding classifier (optional)
├── construct_rule.py           # Graph edge rules
├── alias.py                    # Alias sampling method
├── utils.py                    # Helper functions
│
├── output/                     # Generated outputs
│   ├── r5.2/
│   │   ├── 5/
│   │   │   ├── graph/
│   │   │   ├── sequential_model/
│   │   ├── session_data/
│   │   ├── embedding.pickle
│   │   └── confusion.jpg
```

---

## ⚙️ Pipeline Usage

You can run the pipeline step-by-step or automate it using a `main.py`.

### 1. Preprocess HTTP Data

```bash
python preprocess_http_data.py
```

* Filters blacklisted URLs (e.g., yahoo.com, linkedin.com)
* Samples 10% of other URLs
* Output: `http_process.csv`

---

### 2. Construct Activity Graph

```bash
python graph_construct.py
```

* Reads `logon.csv`, `file.csv`, `http_process.csv`, `device.csv`
* Applies rules to connect activities
* Output: `activity_graph.gpickle`, `activity_graph_edge`

---

### 3. Generate Node Embeddings

```bash
python graph_embedding.py
```

* Choose `Word2Vec` (`use_direct_w2v=True`) or `DeepWalk`
* Output: `embedding.pickle`

---

### 4. Generate Session Data

```bash
python generate_data.py
```

* Groups nodes into time-based sessions
* Labels using CERT answer files
* Output: `sample_session_15`, `sample_session_label_15`

---

### 5. Train Threat Detection Model

```bash
python threat_detection.py
```

* Trains LSTM model on sessions
* Outputs model (`.pth`) and confusion matrix image

---

## 📊 Model Configuration

**In `sequential_model.py`:**

```python
sequence_length = 30
input_size = 128
hidden_size = 256
output_size = 5
```

**In `threat_detection.py`:**

```python
epochs = 1000
batch_size = 4096
loss = CrossEntropyLoss(weight=...)
```

**In `walker.py`:**

```python
walk_length = 30
num_walks = 1
```

---

## 🧪 Troubleshooting & Tips

* **File Not Found**: Check that all CSVs are in the correct path.
* **Memory Error**: Reduce `walk_length` or `num_walks`.
* **Model Accuracy Low**: Try adjusting class weights or increasing training epochs.

🛠 Debug prints (e.g., in `graph_construct.py`, `walker.py`) can help:

```python
print(f"Vertices processed: {len(vertices)}")
print(f"Walk length: {len(walk)}")
```

---

## 📈 Improvements & TODOs

* [ ] Handle sessions with no logon/logoff.
* [ ] Optimize `apply_rule_2/3` using index mapping.
* [ ] Add support for `build_company_graph`, `build_object_graph`.
* [ ] Merge all graphs for a holistic detection model.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For issues, suggestions, or contributions, please open an [Issue](https://github.com/Akobabs/InsiderThreatDetection/issues) or contact the maintainer directly.

---
