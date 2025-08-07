import json
import random
import hashlib
from datetime import datetime, timedelta

def hash_url(url: str) -> str:
    """Generate SHA256 hash of a URL."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def generate_random_features():
    """Generates a dictionary of random URL features."""
    return {
        "qty_dot_url": random.randint(1, 5),
        "qty_hyphen_url": random.randint(0, 3),
        "qty_underline_url": random.randint(0, 2),
        "qty_slash_url": random.randint(1, 6),
        "qty_questionmark_url": random.randint(0, 1),
        "qty_equal_url": random.randint(0, 2),
        "qty_at_url": random.randint(0, 1),
        "qty_and_url": random.randint(0, 2),
        "qty_exclamation_url": random.randint(0, 1),
        "qty_space_url": 0,
        "qty_tilde_url": random.randint(0, 1),
        "qty_comma_url": random.randint(0, 1),
        "qty_plus_url": random.randint(0, 1),
        "qty_asterisk_url": random.randint(0, 1),
        "qty_hashtag_url": random.randint(0, 1),
        "qty_dollar_url": random.randint(0, 1),
        "qty_percent_url": random.randint(0, 1),
        "qty_tld_url": 1,
        "length_url": random.randint(20, 150),
        "email_in_url": random.choice([0, 1]),
    }

def generate_sample_data(num_urls=20, num_predictions=50):
    """Generates sample data for url_info and prediction_history."""
    urls = [
        f"http://example-site-{i}.com/path/to/resource" for i in range(num_urls)
    ] + [
        f"https://secure-service-{i}.net/login?user=test" for i in range(num_urls)
    ] + [
        f"http://phishing-attempt-{i}.org/wp-includes/update.php" for i in range(num_urls)
    ]
    random.shuffle(urls)

    url_info = {}
    prediction_history = {}
    label_names = ['benign', 'defacement', 'malware', 'phishing']

    # Create url_info entries
    for url in urls[:num_urls]:
        url_hash = hash_url(url)
        url_info[url_hash] = {
            'url': url,
            'first_seen_timestamp': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            'extracted_features': generate_random_features(),
            # Optional: Add ground truth labels if available
            'ground_truth': {
                'benign': 1 if 'example-site' in url else 0,
                'defacement': 0,
                'malware': 1 if 'malicious-payload' in url else 0,
                'phishing': 1 if 'phishing-attempt' in url else 0,
            }
        }

    # Create prediction_history entries
    url_hashes = list(url_info.keys())
    for i in range(num_predictions):
        url_hash = random.choice(url_hashes)
        
        # Simulate multi-model predictions
        comparison_results = []
        for model in ['CNN', 'XGB', 'RF']:
            # Simulate label predictions
            labels = {name: 0 for name in label_names}
            malicious_type = random.choice(['defacement', 'malware', 'phishing'])
            if random.random() > 0.6: # 40% chance of being benign
                labels['benign'] = 1
            else:
                labels[malicious_type] = 1

            # Simulate probabilities
            probabilities = [random.random() for _ in range(len(label_names))]
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]

            comparison_results.append({
                'model_name': model,
                'predicted_labels': labels,
                'probabilities': probabilities,
                'execution_time_ms': random.uniform(50, 500)
            })

        prediction_id = f"pred_{i:03d}_{random.randint(1000, 9999)}"
        prediction_history[prediction_id] = {
            'url_hash': url_hash,
            'prediction_timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 14400))).isoformat(),
            'server_timestamp': {'.sv': 'timestamp'},
            'comparison_results': comparison_results
        }

    return {'url_info': url_info, 'prediction_history': prediction_history}

if __name__ == "__main__":
    sample_data = generate_sample_data(num_urls=50, num_predictions=200)
    output_filename = 'sample_data_to_upload.json'

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)

    print(f"Successfully generated sample data in '{output_filename}'")
    print(f"- {len(sample_data['url_info'])} URL info entries created.")
    print(f"- {len(sample_data['prediction_history'])} prediction history entries created.")
