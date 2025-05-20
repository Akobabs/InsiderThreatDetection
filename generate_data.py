from graph_construct import extract_vertices_from_files
from graph_embedding import load_labels
import os
import time
from datetime import datetime

def convert_to_date(time_str):
    """Convert a time string to a datetime object."""
    time_struct = time.strptime(time_str, '%m/%d/%Y %H:%M:%S')
    return datetime(time_struct[0], time_struct[1], time_struct[2])

def calculate_day_difference(time_str1, time_str2):
    """Calculate the number of days between two time strings."""
    date1 = convert_to_date(time_str1)
    date2 = convert_to_date(time_str2)
    return (date2 - date1).days

def process_sessions(vertices, labels, min_day_diff=15):
    """Group vertices into user sessions, label them, and sample based on time differences."""
    user_start_times = {}
    user_last_times = {}

    for vertex in vertices:
        user = vertex['user']
        if user not in user_start_times:
            user_start_times[user] = vertex['timestamp']
        vertex['time_diff'] = vertex['timestamp'] - user_start_times[user]

    sessions = []
    current_session = []
    prev_user = None

    print("Grouping vertices into sessions...")
    for vertex in vertices:
        user = vertex['user']
        if prev_user is None or prev_user == user:
            current_session.append(vertex)
            if vertex['activity'] == 'Logoff':
                sessions.append(current_session)
                current_session = []
        else:
            sessions.append(current_session)
            current_session = [vertex]
        prev_user = user

    if current_session:
        sessions.append(current_session)

    # TODO: Handle sessions without logon/logoff boundaries
    # 1. Sessions starting with logon but no logoff: end at next logon
    # 2. Sessions with logoff but no logon: adjust boundaries

    sessions = sorted(sessions, key=lambda s: s[0]['time_diff'])

    session_labels = []
    session_node_ids = []
    label_counts = {}

    print("Labeling sessions...")
    for session in sessions:
        label_counts_session = [0] * 5
        for node in session:
            label = labels.get(node['id'], 0)
            label_counts_session[label] += 1
        if label_counts_session[0] != len(session):
            if label_counts_session[2] != 0:
                print(f"Label counts: {label_counts_session}")
            label = label_counts_session[1:].index(max(label_counts_session[1:])) + 1
        else:
            label = 0
        session_labels.append(label)
        session_node_ids.append([node['id'] for node in session])
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Total sessions: {len(session_node_ids)}")
    print(f"Label counts: {label_counts}")

    sampled_sessions = []
    sampled_labels = []
    sampled_node_ids = []
    sampled_label_counts = {}

    print("Sampling sessions...")
    for i, session in enumerate(sessions):
        current_time = session[0]['time_str']
        user = session[0]['user']
        if user not in user_last_times:
            user_last_times[user] = current_time
            sampled_sessions.append(session)
            sampled_labels.append(session_labels[i])
            sampled_node_ids.append(session_node_ids[i])
        else:
            previous_time = user_last_times[user]
            same_day = convert_to_date(previous_time) == convert_to_date(current_time)
            day_diff = calculate_day_difference(previous_time, current_time)
            if same_day or day_diff >= min_day_diff or session_labels[i] != 0:
                user_last_times[user] = current_time
                sampled_sessions.append(session)
                sampled_labels.append(session_labels[i])
                sampled_node_ids.append(session_node_ids[i])

    sampled_sessions = sorted(sampled_sessions, key=lambda s: s[0]['time_diff'])

    sampled_label_counts = {}
    for label in sampled_labels:
        sampled_label_counts[label] = sampled_label_counts.get(label, 0) + 1

    print(f"Sampled sessions: {len(sampled_node_ids)}")
    print(f"Sampled label counts: {sampled_label_counts}")

    return sampled_node_ids, sampled_labels

def save_sessions_and_labels(node_ids, labels, output_dir):
    """Save sampled sessions and labels to files."""
    os.makedirs(output_dir, exist_ok=True)
    session_file = os.path.join(output_dir, "sample_session_15")
    label_file = os.path.join(output_dir, "sample_session_label_15")

    print("Saving sampled sessions...")
    with open(session_file, 'w') as file:
        for session in node_ids:
            file.write('\t'.join(session) + '\n')

    print("Saving sampled labels...")
    with open(label_file, 'w') as file:
        for label in labels:
            file.write(f"{label}\n")

if __name__ == '__main__':
    data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
    data_version = "r5.2"
    output_dir = os.path.join("./output", data_version, "session_data")

    vertices = extract_vertices_from_files(os.path.join(data_dir, data_version))
    labels = load_labels(os.path.join(data_dir, "answers"), data_version)
    sampled_node_ids, sampled_labels = process_sessions(vertices, labels)
    save_sessions_and_labels(sampled_node_ids, sampled_labels, output_dir)