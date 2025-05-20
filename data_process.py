import os
from tqdm import tqdm

def preprocess_http_data(data_dir, data_version):
    """Preprocess HTTP data by retaining blacklisted URLs and sampling non-blacklisted ones.

    Args:
        data_dir (str): Directory containing the dataset.
        data_version (str): Version of the dataset (e.g., 'r5.2').

    Writes:
        http_process.csv: Filtered and sampled HTTP entries, excluding the content column.
    """
    # Define blacklisted domains (e.g., job-related or sensitive sites)
    blacklisted_domains = [
        "wikileaks", "yahoo.com", "jobhuntersbible.com", "boeing.com", "linkedin.com",
        "indeed.com", "simplyhired.com", "northropgrumman.com", "aol.com",
        "careerbuilder.com", "raytheon.com", "lockheedmartin.com", "job-hunt.org",
        "craigslist.org", "hp.com", "monster.com"
    ]

    input_path = os.path.join(data_dir, data_version, "http.csv")
    output_path = os.path.join(data_dir, data_version, "http_process.csv")

    print(f"Preprocessing HTTP data from {input_path}...")
    non_blacklisted_count = 0

    with open(output_path, 'w') as output_file:
        with open(input_path, 'r') as input_file:
            for line in tqdm(input_file, desc="Processing HTTP entries"):
                # Parse CSV line
                parts = line.strip().split(',')
                if len(parts) <= 4:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
                    continue

                # Extract and validate URL
                url_parts = parts[4].split('/')
                if len(url_parts) < 3:
                    continue  # Skip invalid URLs

                domain = url_parts[2]  # e.g., 'yahoo.com'

                # Sample non-blacklisted URLs (keep 1/10)
                if domain not in blacklisted_domains:
                    non_blacklisted_count += 1
                    if non_blacklisted_count % 10 != 0:
                        continue  # Skip unless it's the 10th non-blacklisted URL

                # Write line (excluding last column) to output
                output_line = ','.join(parts[:-1])
                output_file.write(output_line + '\n')

    print(f"Preprocessing completed. Output saved to {output_path}")

if __name__ == '__main__':
    data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
    data_version = "r5.2"
    preprocess_http_data(data_dir, data_version)