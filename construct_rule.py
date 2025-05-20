import networkx as nx
from tqdm import tqdm
from itertools import chain
import time
import gc

def timer(function):
    """Decorator to measure and print the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"{function.__name__} started")
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{function.__name__} completed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

@timer
def apply_rule_1(activity_graph, daily_sequences, total_days):
    """Connect nodes for the same user on the same host within the same day."""
    total_days = len(daily_sequences)  # Ensure total_days matches sequence length
    host_to_nodes = [None] * total_days
    edge_count = 0

    start_time = time.time()
    for daily_graph in tqdm(daily_sequences, desc="Processing daily graphs for Rule 1"):
        if daily_graph:
            host_groups = {}
            nodes = list(daily_graph.nodes())
            for node_id in nodes:
                host = daily_graph.nodes[node_id]['host']
                if host not in host_groups:
                    host_groups[host] = [node_id]
                else:
                    host_groups[host].append(node_id)
                    edge_count += 1
                    # Add edge to activity_graph (changed from daily_graph for consistency)
                    activity_graph.add_edge(
                        host_groups[host][-2],
                        host_groups[host][-1],
                        EdgeType=1,
                        weight=1
                    )
            day_index = daily_sequences.index(daily_graph)
            host_to_nodes[day_index] = host_groups

    print(f"Rule 1 execution time: {time.time() - start_time:.2f} seconds")
    print(f"Rule 1 edge count: {edge_count}")
    return activity_graph, host_to_nodes

@timer
def apply_rule_2(activity_graph, daily_sequences, total_days, host_to_nodes):
    """Link behavior chains for the same user on the same host across multiple days."""
    total_days = len(daily_sequences)  # Ensure total_days matches sequence length
    edge_count = 0

    # Note: Commented-out code to merge daily graphs is omitted as it's unused
    for day_i in range(total_days):
        for day_j in range(day_i + 1, total_days):
            if not (daily_sequences[day_i] and daily_sequences[day_j]):
                continue
            if host_to_nodes[day_i] and host_to_nodes[day_j]:
                for host in host_to_nodes[day_i]:
                    if host in host_to_nodes[day_j]:
                        # TODO: Consider adding logon -> logoff as a Rule 3 pattern
                        start_i = host_to_nodes[day_i][host][0]
                        end_i = host_to_nodes[day_i][host][-1]
                        start_j = host_to_nodes[day_j][host][0]
                        end_j = host_to_nodes[day_j][host][-1]

                        count_i = len(host_to_nodes[day_i][host])
                        count_j = len(host_to_nodes[day_j][host])
                        weight = count_i / count_j if count_i < count_j else count_j / count_i

                        activity_graph.add_edge(start_i, start_j, EdgeType=2, weight=weight)
                        activity_graph.add_edge(end_i, end_j, EdgeType=2, weight=weight)
                        edge_count += 2

    # Note: Commented-out memory cleanup is retained but unused
    # del daily_sequences
    # gc.collect()

    print(f"Rule 2 edge count: {edge_count}")
    return activity_graph

@timer
def apply_rule_3(activity_graph, daily_sequences, total_days, host_to_nodes):
    """Associate group operation types across days for the same user and host."""
    edge_count = 0
    operation_patterns = [["File Open", "File Write"], ["Connect", "Disconnect"]]
    valid_operations = list(chain.from_iterable(operation_patterns))

    def identify_pattern_matches(node_ids):
        """Identify nodes matching group operation patterns for the same object."""
        activity_map = {}
        for node_id in node_ids:
            act_type = activity_graph.nodes[node_id]['activity']
            if act_type in valid_operations:
                if act_type not in activity_map:
                    activity_map[act_type] = []
                activity_map[act_type].append([node_id, 0])  # 0 = unmatched

        for pattern in operation_patterns:
            if pattern[0] in activity_map and pattern[1] in activity_map:
                for start_node in activity_map[pattern[0]]:
                    obj = activity_graph.nodes[start_node[0]]['obj']
                    for end_node in activity_map[pattern[-1]]:
                        if obj == activity_graph.nodes[end_node[0]]['obj']:
                            start_node[1] = end_node[1] = 1  # Mark as matched
                            break

        # Remove unmatched nodes
        for act_type in list(activity_map.keys()):
            activity_map[act_type] = [node for node in activity_map[act_type] if node[1] == 1]
            if not activity_map[act_type]:
                del activity_map[act_type]

        return activity_map

    host_day_activities = {}
    host_day_patterns = {}
    for day_idx in range(total_days):
        if host_to_nodes[day_idx] is None:
            continue
        for host in host_to_nodes[day_idx]:
            if host not in host_day_activities:
                host_day_activities[host] = []
                host_day_patterns[host] = []
            host_day_activities[host].append(host_to_nodes[day_idx][host])
            host_day_patterns[host].append(identify_pattern_matches(host_to_nodes[day_idx][host]))

    print("Starting edge construction for Rule 3")
    for host in tqdm(host_day_patterns, desc="Building Rule 3 edges"):
        for day_i in range(len(host_day_patterns[host])):
            for day_j in range(day_i + 1, len(host_day_patterns[host])):
                for activity in host_day_patterns[host][day_i]:
                    if activity in host_day_patterns[host][day_j]:
                        for node_i in host_day_patterns[host][day_i][activity]:
                            for node_j in host_day_patterns[host][day_j][activity]:
                                # TODO: Define a dynamic weight
                                activity_graph.add_edge(node_i[0], node_j[0], EdgeType=3, weight=0.5)
                                edge_count += 1

    print(f"Rule 3 edge count: {edge_count}")
    return activity_graph

@timer
def apply_rule_3_1(activity_graph, daily_sequences, total_days, host_to_nodes):
    """Connect nodes with the same operation type on the same host within and across days."""
    total_days = len(daily_sequences)  # Ensure total_days matches sequence length
    print("Starting Rule 3.1: Same day and cross-day same operation type connections")

    print("Building same-day, same-host, same-operation-type connections")
    operation_to_nodes = [None] * total_days
    for daily_graph in daily_sequences:
        if daily_graph:
            day_idx = daily_sequences.index(daily_graph)
            host_groups = host_to_nodes[day_idx]
            operation_groups = {}

            for host in host_groups:
                operation_map = {}
                for node_id in host_groups[host]:
                    operation = daily_graph.nodes[node_id]['activity']
                    if operation not in operation_map:
                        operation_map[operation] = [node_id]
                    else:
                        # Add edge for same user, same day, same host, same operation
                        activity_graph.add_edge(
                            operation_map[operation][-1],
                            node_id,
                            EdgeType=3,
                            weight=1
                        )
                        operation_map[operation].append(node_id)
                operation_groups[host] = operation_map

            operation_to_nodes[day_idx] = operation_groups

    print("Completed same-day, same-host, same-operation-type connections")

    print("Building cross-day, same-host, same-operation-type connections")
    for day_i in range(total_days):
        for day_j in range(day_i + 1, total_days):
            if daily_sequences[day_i] and daily_sequences[day_j]:
                for host in host_to_nodes[day_i]:
                    if host in host_to_nodes[day_j]:
                        if operation_to_nodes[day_i] and operation_to_nodes[day_j]:
                            for operation in operation_to_nodes[day_i].get(host, {}):
                                if operation in operation_to_nodes[day_j].get(host, {}):
                                    start_i = operation_to_nodes[day_i][host][operation][0]
                                    start_j = operation_to_nodes[day_j][host][operation][0]
                                    end_i = operation_to_nodes[day_i][host][operation][-1]
                                    end_j = operation_to_nodes[day_j][host][operation][-1]

                                    count_i = len(operation_to_nodes[day_i][host][operation])
                                    count_j = len(operation_to_nodes[day_j][host][operation])
                                    weight = round(min(count_i, count_j) / max(count_i, count_j), 3)

                                    activity_graph.add_edge(start_i, start_j, EdgeType=3, weight=weight)
                                    activity_graph.add_edge(end_i, end_j, EdgeType=3, weight=weight)

    print("Completed cross-day, same-host, same-operation-type connections")
    print("Rule 3.1 completed")
    return activity_graph, operation_to_nodes