import numpy as np

def create_alias_table(probabilities):
    """Create alias table for efficient sampling from a discrete distribution.

    Args:
        probabilities (list): List of probabilities summing to 1.

    Returns:
        tuple: (accept_probabilities, alias_indices) for sampling.
    """
    num_elements = len(probabilities)
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1")

    accept_probabilities = [0] * num_elements
    alias_indices = [0] * num_elements
    small_queue = []
    large_queue = []
    scaled_probabilities = np.array(probabilities) * num_elements

    for index, prob in enumerate(scaled_probabilities):
        if prob < 1.0:
            small_queue.append(index)
        else:
            large_queue.append(index)

    while small_queue and large_queue:
        small_index = small_queue.pop()
        large_index = large_queue.pop()
        accept_probabilities[small_index] = scaled_probabilities[small_index]
        alias_indices[small_index] = large_index
        scaled_probabilities[large_index] -= (1 - scaled_probabilities[small_index])

        if scaled_probabilities[large_index] < 1.0:
            small_queue.append(large_index)
        else:
            large_queue.append(large_index)

    while large_queue:
        accept_probabilities[large_queue.pop()] = 1.0
    while small_queue:
        accept_probabilities[small_queue.pop()] = 1.0

    return accept_probabilities, alias_indices

def sample_from_alias(accept_probabilities, alias_indices):
    """Sample an index using the alias method.

    Args:
        accept_probabilities (list): Acceptance probabilities from alias table.
        alias_indices (list): Alias indices from alias table.

    Returns:
        int: Sampled index.
    """
    num_elements = len(accept_probabilities)
    if num_elements == 0:
        raise ValueError("Alias table is empty")

    index = int(np.random.random() * num_elements)
    threshold = np.random.random()
    return index if threshold < accept_probabilities[index] else alias_indices[index]