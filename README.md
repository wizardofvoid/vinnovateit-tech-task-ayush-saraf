## Academic Success Predictor

A Streamlit-based machine learning app that predicts whether a student will pass or fail based on academic metrics like attendance, midterm scores, assignment performance, and more. Enhanced with LLaMA-based AI explanations using Together AI's API.

## Environment Setup:

python Version: 3.12 or higher (latest version recommended)

For a demo view you can visit this link:
https://vinnovateit-tech-task-ayush-saraf-mljuhuuahuulfr5i4appgud.streamlit.app/

## To run locally:
1. Clone the repository:

    ```bash 
        git clone https://github.com/wizardofvoid/vinnovateit-tech-task-ayush-saraf    
        cd vinnovateit-tech-task-ayush-saraf
    ```

2. Install dependencies:

    ```bash
        pip install -r requirements.txt
    ```
    or install manually:

    ```bash
        pip install streamlit scikit-learn numpy together
    ```

## How to run the App:

Open the terminal, go to the code directory and type:

```bash
streamlit run app.py
```

The app will launch in the browser using the localhost. You should automatically be directed to the webstite

## Features:

- Predicts student academic outcome (Pass/Fail)

- Uses a trained Random Forest model

- Generates AI explanations using LLaMA 2 via Together AI

- Interactive sliders for input features

## Optional: If you want to see how the model was trained:

1. Install jupyter notebook using:

    ```bash
        pip install jupyter
    ```

2. Open jupyter notebook using:

    ```bash
        jupyter notebook
    ```

3. Open `Task3.ipynb` from the file directory shown in the jupyter GUI