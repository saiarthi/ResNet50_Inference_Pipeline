# ResNet50_Inference_Pipeline
This project aims to compare the inference outputs of two ResNet50 models: one from the torchvision library and a custom-trained local model. The comparison focuses on evaluating the precision of the outputs using the Pearson Correlation Coefficient (PCC) and identifying discrepancies in the preprocessing steps.

# Requirements
Python 3.10 or later - (Recommended)
PyTorch
torchvision
PIL (Python Imaging Library)
requests
numpy

# Installation
# 1. Clone the repository:
```bash
git clone https://github.com/saiarthi/ResNet50_Inference_Pipeline.git
cd resnet50-inference-comparison
```

# 2.Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

# 3.Install the required packages:
```bash
Copy code
pip install -r requirements.txt
```

# Project Structure
```bash
resnet50-inference-comparison/
│
├── compare_models.py              # Script to compare the outputs of both models
├── comp_pcc.py                    # Pearson Correlation Coefficient computation
├── load_image.py                  # Image loading and preprocessing utilities
├── local_model_inference.py       # Local model inference script
├── reference_model_inference.py   # Reference model inference script
├── resnet50.py                    # Custom ResNet50 model definition
├── requirements.txt               # Project dependencies
├── README.md                      # Project readme
```

# Usage
# Run the Comparison:
```bash
python compare_models.py
```

# Check Outputs:
The script will print the predicted class indices and names for both models, along with the comparison result and the Pearson Correlation Coefficient (PCC).
