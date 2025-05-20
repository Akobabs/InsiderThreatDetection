def preprocess_graph(graph):
    """Map graph nodes to indices for efficient processing.

    Args:
        graph: NetworkX graph.

    Returns:
        tuple: (node_to_index, index_to_node) mappings.
    """
    node_to_index = {}
    index_to_node = []
    for index, node in enumerate(graph.nodes()):
        node_to_index[node] = index
        index_to_node.append(node)
    return index_to_node, node_to_index

def partition_dictionary(items, num_workers):
    """Partition a dictionary into chunks for parallel processing."""
    batch_size = (len(items) - 1) // num_workers + 1
    partitions = []
    current_partition = []
    count = 0
    for key, value in items.items():
        current_partition.append((key, value))
        count += 1
        if count % batch_size == 0:
            partitions.append(current_partition)
            current_partition = []
    if current_partition:
        partitions.append(current_partition)
    return partitions

def partition_list(items, num_workers):
    """Partition a list into chunks for parallel processing."""
    batch_size = (len(items) - 1) // num_workers + 1
    partitions = []
    current_partition = []
    count = 0
    for index, item in enumerate(items):
        current_partition.append((index, item))
        count += 1
        if count % batch_size == 0:
            partitions.append(current_partition)
            current_partition = []
    if current_partition:
        partitions.append(current_partition)
    return partitions

def partition_number(total, num_workers):
    """Divide a number into parts for parallel processing."""
    if total % num_workers == 0:
        return [total // num_workers] * num_workers
    return [total // num_workers] * (num_workers - 1) + [total % num_workers]