A FastAPI application that demonstrates FGSM (Fast Gradient Sign Method) adversarial attacks on a pre-trained MNIST digit classifier. This API allows you to generate adversarial examples and test the robustness of your model against FGSM attacks.

Features
MNIST Classification: Pre-trained neural network for handwritten digit recognition
FGSM Attack: Generate adversarial examples using the FGSM algorithm
REST API: Simple HTTP endpoints for interacting with the model
Image Processing: Automatic image resizing and normalization for inputs

Before running the app, make sure you have:
Python 3.8+ (Recommended: 3.10 or 3.11)
pip (Python package manager)
Git (optional, for cloning the repository)

STEPS TO FOLLOW:
1. Clone repository
2. Create virtual python environment (python -m venv venv
venv\Scripts\activate
)
3. python3 -m venv venv (to enter environment)
4. pip install -r requirements.txt
5. Run the server locally uvicorn app:app --reload --host 0.0.0.0 --port 8000
6. Download the flutter APK and give a MNIST image and select an epsilon value and click process image
7. You should see the response 200ok if everything is working as intended and the card will appear to show the results(attack status) returned by /attack endpoint
