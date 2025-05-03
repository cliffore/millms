import argparse
import requests
import sys
import re


def extract_from_path(data, path):
    parts = re.split(r'\.(?![^\[]*\])', path)  # split on '.' but ignore inside brackets
    for part in parts:
        # Handle list indexing: e.g., explanations[0]
        match = re.match(r'([^\[]+)(\[(\d+)\])?', part)
        if not match:
            raise ValueError(f"Invalid path segment: {part}")
        key = match.group(1)
        index = match.group(3)

        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            raise KeyError(f"Key '{key}' not found")

        if index is not None:
            index = int(index)
            if isinstance(data, list) and 0 <= index < len(data):
                data = data[index]
            else:
                raise IndexError(f"Index [{index}] out of bounds for key '{key}'")
    return data


def main():

    base_url = "https://www.neuronpedia.org/api/feature/gemma-2-2b/"

    parser = argparse.ArgumentParser(description="Send a request to an API and extract a JSON field.")    
    parser.add_argument('--layer', type=int, help='The layer number to use')
    parser.add_argument('--feature', type=int, help='The feature id to use')
    args = parser.parse_args()

    layer_to_use = str(args.layer) + '-gemmascope-res-16k'
    feature_to_use = int(args.feature)
    print("Layer to use:", layer_to_use)

    args = parser.parse_args()

    request_url = base_url + str(layer_to_use) + "/" + str(feature_to_use)

    # Send GET request
    try:
        response = requests.get(request_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON response
    try:
        data = response.json()
        #print(data)
    except ValueError:
        print("Response was not valid JSON.", file=sys.stderr)
        sys.exit(1)


    try:
        value = extract_from_path(data, 'explanations[0].description')
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error extracting key path '{args.key}': {e}", file=sys.stderr)
        sys.exit(1)

    print(value)


if __name__ == '__main__':
    main()