import requests
import base64
import io
import random
from torchvision.datasets import FashionMNIST
from fmnist_inference_service import FMNIST_LABELS
from fmnist_dataset import download_dataset
from pathlib import Path
from server_config import ServerConfig

server_config = ServerConfig().config_data

def run_client_example():
    """
    Selects a random image from the FMNIST test split, prints its label, sends it to the
    inference service, and prints the predicted label.
    """
    download_dataset(Path(__file__).parent / "dataset")
    test_dataset = FashionMNIST(root="dataset/", train=False, download=False)

    random_index = random.randint(0, len(test_dataset) - 1)
    image, label_idx = test_dataset[random_index]
    true_label = FMNIST_LABELS[label_idx]

    print(f"Sending Image: #{random_index}")
    print(f"True Label: {true_label}")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    service_url = f"http://{server_config['host']}:{server_config['port']}/predict"
    payload = {"image_base64": img_base64}

    try:
        response = requests.post(service_url, json=payload)
        response.raise_for_status()

        result = response.json()
        print("\nServer Response")
        print(f"Predicted Label: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")

    except requests.exceptions.RequestException as e:
        print(f"Inference service request error: {e}")

if __name__ == "__main__":
    run_client_example()