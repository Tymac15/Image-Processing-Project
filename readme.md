<h2 align ='center'> COS791- Image Processing Final Project </h3>
<p align = 'center'>Ball Tracking Software</p>


<img src="readme_resources/bar.png" alt="Alt text" title="bar">

## Description

This project required the creation of a piece of software that would take a video as input and perform the relevant processing required to identify and track a ball/puck within a field of play.
## Documentation
<div><a href="https://www.overleaf.com/read/mbbcfmpmfktc#854a0b">🧾 Documentation</a></div>
<div><a href="https://drive.google.com/file/d/1jkOIvxXDZvI2KDQqlkBXQ5gFo0clQ3yN/view?usp=sharing">📽️ Demo Video </a></div>
<div><a href="[https://drive.google.com/file/d/1jkOIvxXDZvI2KDQqlkBXQ5gFo0clQ3yN/view?usp=sharing](https://drive.google.com/file/d/1436jDJVtvuZ7ax8HtKpQHgg4jj8iuI-a/view?usp=sharing)">📽️ Field hockey output </a></div>
<div><a href="[https://drive.google.com/file/d/1jkOIvxXDZvI2KDQqlkBXQ5gFo0clQ3yN/view?usp=sharing](https://drive.google.com/file/d/17OrBAij6Sdn6PP3A0mAU9h2LzvllaKpw/view?usp=sharing)">📽️ Ice hockey output </a></div>

<img src="readme_resources/bar.png" alt="Alt text" title="bar">


## Training a Model with YOLOv8 on Google Colab

Follow these steps to set up and train a YOLOv8 model using Google Colab:

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/) and log in with your Google account.

2. **Upload the Jupyter Notebook**
   - Download the Jupyter notebook file (`google_colab_yolov8.ipynb`) from this repository to your local machine.
   - In Colab, select **File > Upload Notebook** and choose your downloaded `.ipynb` file.

3. **Connect Google Drive**
   - To store your datasets, model weights, or output files, you’ll need to connect Google Drive:
   - Add and run the following code in the first cell of your notebook:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Follow the authorization steps provided to link your Google Drive with Colab.

4.  **Data file structure**
 ```python
MyDrive
├── dataset
│   ├── field_hockey
│   │   ├── phase_1
│   │   └── phase_2
│   └── ice_hockey
│       ├── phase_1
│       └── phase_2
└── models
    ├── field_hockey
    └── ice_hockey
```

5. **Install YOLOv8 Dependencies**
   - Ensure you have all necessary dependencies installed, especially `ultralytics`, which provides the YOLOv8 model. Use the following command:
     ```python
     !pip install ultralytics
     ```

6. **Organize Files and Dataset**
   - Place any dataset files or folders (e.g., images and labels) into your Google Drive. Make sure the path in your notebook code corresponds to the dataset location in your Google Drive.

7. **Run Each Cell Sequentially**
   - Execute each cell in the notebook in sequence. The notebook follows this structure:
     - **Environment Setup**: Check GPU availability and set the working directory.
     - **Google Drive Connection**: Mount Google Drive to save and retrieve files.
     - **Install Dependencies**: Install the `ultralytics` package for YOLOv8.
     - **Set Parameters**: Define variables for your dataset type (e.g., `field_hockey`) and experiment name.
     - **Training (Two Phases)**:
       - **Phase 1**: Load a pre-trained YOLOv8 model, fine-tune it on a large dataset, and validate. Save the fine-tuned model to Google Drive.
       - **Phase 2**: Load the Phase 1 model, further fine-tune it on a smaller dataset, validate, and save the model.
     - **Results Access**: After training, check your Google Drive folder for saved models and training logs.


8. **Accessing Results**
   - After training, check your Google Drive folder for outputs, such as saved model weights and evaluation metrics. Most importantly, the phase_2_01.pt file will be necessary to run the ball tracking script. This file can be following the models directory.


<img src="readme_resources/bar.png" alt="Alt text" title="bar">

## Running the Code
To run the ball tracking software, go into main.py and check that the path to the input video is correct. Additionally, check that the path to the model (the .pt file obtained from step 8. described in the model training section) is correct and then run the script.








