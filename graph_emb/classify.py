import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class TopKRanker(OneVsRestClassifier):
    """Classifier that predicts top-k labels based on probability scores."""
    def predict(self, features, top_k_list):
        probabilities = np.asarray(super().predict_proba(features))
        all_labels = []
        for index, k in enumerate(top_k_list):
            prob = probabilities[index, :]
            top_indices = prob.argsort()[-k:]
            labels = self.classes_[top_indices].tolist()
            binary_labels = np.zeros_like(prob)
            binary_labels[labels] = 1
            all_labels.append(binary_labels)
        return np.asarray(all_labels)

class GraphClassifier:
    """Classifier for graph nodes using embeddings and a top-k ranker.

    Args:
        embeddings (dict): Mapping of node IDs to embedding vectors.
        classifier: Underlying classifier (e.g., LogisticRegression).
    """
    def __init__(self, embeddings, classifier):
        self.embeddings = embeddings
        self.classifier = TopKRanker(classifier)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, node_ids, labels, all_labels):
        """Train the classifier on node embeddings and labels."""
        self.binarizer.fit(all_labels)
        features = [self.embeddings[node_id] for node_id in node_ids if node_id in self.embeddings]
        if len(features) != len(node_ids):
            print(f"Warning: {len(node_ids) - len(features)} nodes lack embeddings")
        binary_labels = self.binarizer.transform(labels)
        self.classifier.fit(features, binary_labels)

    def evaluate(self, node_ids, labels):
        """Evaluate the classifier on test data.

        Returns:
            dict: F1 scores (micro, macro, samples, weighted) and accuracy.
        """
        top_k_list = [len(label_set) for label_set in labels]
        predicted_labels = self.predict(node_ids, top_k_list)
        true_labels = self.binarizer.transform(labels)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {
            average: f1_score(true_labels, predicted_labels, average=average)
            for average in averages
        }
        results['accuracy'] = accuracy_score(true_labels, predicted_labels)
        print("Evaluation Results:")
        print(results)
        return results

    def predict(self, node_ids, top_k_list):
        """Predict labels for given nodes."""
        features = [self.embeddings[node_id] for node_id in node_ids if node_id in self.embeddings]
        if len(features) != len(node_ids):
            print(f"Warning: {len(node_ids) - len(features)} nodes lack embeddings")
        return self.classifier.predict(np.asarray(features), top_k_list)

    def split_train_evaluate(self, node_ids, labels, train_ratio, seed=0):
        """Split data, train, and evaluate the classifier."""
        state = np.random.get_state()
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(len(node_ids))
        train_size = int(train_ratio * len(node_ids))

        train_nodes = [node_ids[i] for i in shuffle_indices[:train_size]]
        train_labels = [labels[i] for i in shuffle_indices[:train_size]]
        test_nodes = [node_ids[i] for i in shuffle_indices[train_size:]]
        test_labels = [labels[i] for i in shuffle_indices[train_size:]]

        self.train(train_nodes, train_labels, labels)
        np.random.set_state(state)
        return self.evaluate(test_nodes, test_labels)

def read_node_labels(filename, skip_header=False):
    """Read node IDs and labels from a file.

    Args:
        filename (str): Path to the label file.
        skip_header (bool): Whether to skip the first line.

    Returns:
        tuple: (node_ids, labels) where labels are lists of label strings.
    """
    node_ids = []
    labels = []
    with open(filename, 'r') as file:
        if skip_header:
            file.readline()
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(' ')
            if len(parts) < 1:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue
            node_ids.append(parts[0])
            labels.append(parts[1:])
    return node_ids, labels