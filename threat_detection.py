import os
import torch
import torch.utils.data as Data
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sequential_model import SequentialModel

def normalize_session_embedding(session_embedding, max_length=30, embedding_size=128):
    """Pad or truncate session embeddings to a fixed length."""
    zero_embedding = [0] * embedding_size
    if len(session_embedding) > max_length:
        return session_embedding[:max_length]
    return session_embedding + [zero_embedding] * (max_length - len(session_embedding))

def load_node_embeddings(embedding_dir):
    """Load node embeddings from a pickle file."""
    print(f"Loading embeddings from {embedding_dir}...")
    with open(embedding_dir, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

def load_session_data(data_dir, data_version, version, batch_size=64):
    """Load session embeddings and labels, split into train/test datasets."""
    print("Loading session data...")
    start_time = time.time()

    embedding_dir = f"/home/zhengchaofan/project/ltc_insider/output/{data_version}/{version}end-embedding.pickle"
    embeddings = load_node_embeddings(embedding_dir)
    zero_embedding = [0] * 128

    session_embeddings = []
    session_labels = []
    has_embedding = 0
    no_embedding = 0

    with open(os.path.join(data_dir, "sample_session"), 'r') as file:
        for line in tqdm(file, desc="Reading sessions"):
            session_embedding = []
            node_ids = line.strip().split('\t')
            for node_id in node_ids:
                if node_id in embeddings:
                    session_embedding.append(embeddings[node_id].tolist())
                    has_embedding += 1
                else:
                    session_embedding.append(zero_embedding)
                    no_embedding += 1
            session_embedding = normalize_session_embedding(session_embedding)
            session_embeddings.append(session_embedding)

    print(f"Nodes with embeddings: {has_embedding}, without embeddings: {no_embedding}")

    with open(os.path.join(data_dir, "sample_session_label"), 'r') as file:
        session_labels = [int(line.strip()) for line in file]

    assert len(session_embeddings) == len(session_labels), "Mismatch between embeddings and labels"

    session_embeddings = torch.tensor(session_embeddings, dtype=torch.float)
    session_labels = torch.tensor(session_labels, dtype=torch.long)

    train_ratio = 0.7
    split_index = int(len(session_embeddings) * train_ratio)
    train_dataset = Data.TensorDataset(session_embeddings[:split_index], session_labels[:split_index])
    test_dataset = Data.TensorDataset(session_embeddings[split_index:], session_labels[split_index:])

    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False)

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    return train_dataset, test_dataset, train_loader, test_loader

def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, and F1 score for each label."""
    metrics = {}
    total_f1 = 0
    for label in range(len(confusion_matrix)):
        metrics[label] = {}
        row_sum = sum(confusion_matrix[label])
        col_sum = sum(confusion_matrix[:, label])
        metrics[label]["recall"] = confusion_matrix[label][label] / row_sum if row_sum > 0 else 0
        metrics[label]["precision"] = confusion_matrix[label][label] / col_sum if col_sum > 0 else 0
        if metrics[label]["precision"] + metrics[label]["recall"] == 0:
            metrics[label]["f1"] = 0
        else:
            metrics[label]["f1"] = 2 * metrics[label]["precision"] * metrics[label]["recall"] / (
                metrics[label]["precision"] + metrics[label]["recall"]
            )
        total_f1 += metrics[label]["f1"]
        print(f"Label {label}: Recall={metrics[label]['recall']:.2f}, Precision={metrics[label]['precision']:.2f}, F1={metrics[label]['f1']:.2f}")
    return total_f1 / len(confusion_matrix)

def plot_confusion_matrix(confusion_matrix, labels, title):
    """Plot a normalized confusion matrix."""
    print("Generating confusion matrix plot...")
    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(normalized_cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("./output/confusion.jpg")
    plt.close()

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset."""
    model.eval()
    predicted_labels = []
    predicted_scores = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating model"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted_scores.append(outputs.cpu().numpy() / outputs.sum(dim=1, keepdim=True).cpu().numpy())
            predicted_labels_batch = torch.argmax(outputs, dim=-1)
            predicted_labels.append(predicted_labels_batch.cpu())
            true_labels.append(labels.cpu())

    predicted_labels = torch.cat(predicted_labels).numpy()
    true_labels = torch.cat(true_labels).numpy()
    confusion = confusion_matrix(true_labels, predicted_labels)

    print("Confusion Matrix:")
    print(confusion)

    macro_f1 = calculate_metrics(confusion)

    onehot_encoder = OneHotEncoder(sparse=False)
    true_encoded = onehot_encoder.fit_transform(true_labels.reshape(-1, 1))
    pred_encoded = onehot_encoder.transform(predicted_labels.reshape(-1, 1))
    auc = roc_auc_score(true_encoded, pred_encoded, multi_class='ovr')
    print(f"AUC: {auc:.4f}")

    return macro_f1

if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_version = "r5.2"
    version = "5"
    data_dir = f"./output/{data_version}/session_data/"
    model_save_path = f"./output/{data_version}/{version}/sequential_model/"
    os.makedirs(model_save_path, exist_ok=True)

    epochs = 1000
    batch_size = 4096

    train_dataset, test_dataset, train_loader, test_loader = load_session_data(data_dir, data_version, version, batch_size)
    model = SequentialModel(
        sequence_length=30,
        input_size=128,
        hidden_size=256,
        output_size=5,
        batch_first=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 40, 20, 40, 20], dtype=torch.float)).to(device)

    model_file = os.path.join(model_save_path, "model_sample.pth")
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        model.load_state_dict(torch.load(model_file))
        model.to(device)

    best_macro_f1 = 0
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.5f}, Time: {time.time() - start_time:.2f} seconds")
        if (epoch + 1) % 2 == 0:
            macro_f1 = evaluate_model(model, test_loader, device)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                print(f"New best F1: {macro_f1:.4f}")
                print(f"Saving model to {model_file}")
                torch.save(model.cpu().state_dict(), model_file)
                model.to(device)
            print("=" * 20)
    