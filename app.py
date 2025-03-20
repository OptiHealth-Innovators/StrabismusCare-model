from flask import Flask, request, jsonify
import detect_strabismus as detector
import torch
import io
from PIL import Image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has images
        if 'images' not in request.files:
            return jsonify({'error': 'No images in the request'}), 400

        files = request.files.getlist('images')

        if len(files) == 0:
            return jsonify({'error': 'No images selected'}), 400

        # Process each image and collect probabilities
        all_probs = []
        individual_results = []

        for file in files:
            try:
                # Read image data
                img_bytes = file.read()

                # Preprocess image with visualization data
                img_tensor = detector.preprocess_image_from_bytes(img_bytes, return_visualization=False)

                if img_tensor is None:
                    continue

                # Get probabilities
                probs = detector.get_probabilities(img_tensor)
                all_probs.append(probs)

                # Get individual result
                _, predicted = torch.max(probs, 1)
                class_idx = predicted.item()

                # Extract the probability for the predicted class
                prob_value = probs[0][class_idx].item()

                # Add EAR values to the result if available
                ear_info = {}
                if "ear_values" in img_tensor:
                    ear_info = img_tensor["ear_values"]

                individual_results.append({
                    'filename': file.filename,
                    'prediction': detector.CLASS_NAMES[class_idx],
                    'class_index': class_idx,
                    'confidence': float(prob_value * 100),
                    'ear_values': ear_info
                })

            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
                individual_results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
                continue

        if not all_probs:
            return jsonify({'error': 'No images were successfully processed'}), 400

        # Stack all probability tensors and calculate the mean
        all_probs = torch.cat(all_probs, dim=0)
        avg_probs = torch.mean(all_probs, dim=0)

        # Get the predicted class from averaged probabilities
        _, predicted = torch.max(avg_probs.unsqueeze(0), 1)
        predicted_class_idx = predicted.item()
        predicted_class = detector.CLASS_NAMES[predicted_class_idx]

        # Prepare response
        response = {
            'overall_result': {
                'prediction': predicted_class,
                'class_index': predicted_class_idx,
                'confidence': float(avg_probs[predicted_class_idx].item() * 100),
                'probabilities': avg_probs.numpy().tolist()
            },
            'individual_results': individual_results
        }

        return jsonify(response)


if __name__ == '__main__':
    # Load the model before starting the server
    detector.initialize_model('vit_epoch.pth')
    print(f"Model loaded and running on {detector.DEVICE}")
    # Run the server
    app.run(host='0.0.0.0', port=1024)
