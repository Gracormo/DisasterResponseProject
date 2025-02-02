# Project Disaster Response Pipeline  

## Project Overview
The **Project Disaster Response Pipeline** is a data engineering and machine learning project focused on **building an end-to-end pipeline** for processing disaster-related messages. The goal is to classify messages sent during disasters so that **emergency responders** can quickly understand their nature and respond effectively.

The project covers:  
- **ETL Pipeline:** Extract, clean, and store messages in a SQLite database.  
- **Machine Learning Pipeline:** Train and evaluate a multi-label text classification model.  
- **Web Application Deployment:** Build a Flask-based app for real-time message classification.

This project is part of Udacity's Data Science Nanodegree and focuses on building a pipeline for text processing, model training, and deployment as a web application. The dataset used for this project was provided by **Figure Eight** and contains real-world messages from disaster scenarios. 

---

## Installation
The code in this repository requires standard Python libraries. Ensure you have **Python 3.8+** installed and run:  

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies.

To set up the project, follow these steps:

### Clone the Repository  
```bash
git clone https://github.com/Gracormo/DisasterResponseProject.git
cd DisasterResponseProject
```

### Create and Activate a Virtual Environment  
```bash
python3 -m venv env  # Create a virtual environment
source env/bin/activate  # Activate it (Mac/Linux)
# For Windows use: env\Scripts\activate
```

### Install Dependencies  
```bash
pip install -r requirements.txt
```

### Run the ETL Pipeline (Extract, Transform, Load)  
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### Train the Machine Learning Model  
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### Run the Web Application  
```bash
python app/run.py
# Open http://127.0.0.1:3000/ in your browser
```

---

## File Descriptions
This repository is structured as follows:

### **1. Web Application (`app/`)**  
- `app/run.py` → **Flask web application** to classify user messages.  
- `app/templates/master.html` → **Main web UI template** for displaying classification results.  
- `app/templates/go.html` → **Web UI** for presenting categorized messages.

### **2. Data Processing (`data/`)**  
- `data/disaster_messages.csv` → **Raw messages dataset** collected from disaster scenarios.  
- `data/disaster_categories.csv` → **Categories dataset** mapping messages to classification labels.  
- `data/DisasterResponse.db` → **SQLite database** containing cleaned disaster messages.  
- `data/process_data.py` → **ETL pipeline script** for data cleaning and storage.

### **3. Machine Learning (`models/`)**  
- `models/train_classifier.py` → **Script to train a multi-label text classifier** using NLP.
- `models/classifier.pkl` → Trained machine learning model (not included due to size constraints).

#### **Pipeline Preparation (`models/preparation/`)**  
- `models/preparation/ETL Pipeline Preparation.ipynb` → **Jupyter Notebook** for ETL pipeline experimentation and preprocessing.  
- `models/preparation/ML Pipeline Preparation.ipynb` → **Jupyter Notebook** for model experimentation and tuning.

### **4. Other Files**  
- `LICENSE` → **Project license** and dataset usage terms.  
- `README.md` → **Project documentation**.  
- `requirements.txt` → **List of required Python libraries**.

---

## Results
The trained **NLP-based classifier** categorizes messages into **36 disaster response categories** with reasonable accuracy. Some key findings:

- **Most common categories:**  
  - `related`, `request`, `offer`, `aid_related`, `weather_related`.  
- **Challenges with imbalanced data:**  
  - Categories like `missing_people` and `refugees` had limited training samples, affecting model performance.  
- **Model Performance:**  
  - A **Random Forest Classifier** combined with **TF-IDF** and **GridSearchCV** was used for optimization.

To see these results in action, visit the **web app** and test a disaster-related message!

---

## Licensing, Authors, Acknowledgements <a name="licensing"></a>  
### **Licensing:**  
- The dataset was provided by **Figure Eight** as part of Udacity’s Data Science Nanodegree.  
- For more details on dataset licensing, visit [Figure Eight](https://www.figure-eight.com/).  

### **Authors:**  
- Developed by **Graco** for the **Project Disaster Response Pipeline** as part of Udacity’s Data Science Nanodegree.  
- Connect with me on **[LinkedIn](https://www.linkedin.com/in/gracorabello/)**.  

### **Acknowledgements:**  
Special thanks to **Udacity** and **Figure Eight** for providing the dataset and structured learning path.

---

**Feel free to use this project as a reference or for learning purposes!** 🚀
