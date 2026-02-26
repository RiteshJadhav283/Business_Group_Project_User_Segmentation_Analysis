# Board Game User Segmentation & Rating-Driven Revenue Analysis

This project performs a comprehensive analysis of the BoardGameGeek dataset, focusing on segmenting users and understanding the economic drivers of board game popularity.

## Project Overview

The analysis is divided into several key phases:
1.  **Data Cleaning & Preprocessing**: Handling large-scale review data and game metadata.
2.  **Exploratory Data Analysis (EDA)**: Visualizing rating distributions and market penetration.
3.  **User Segmentation**: Using K-Means clustering to identify distinct user personas based on engagement and sentiment.
4.  **Revenue Impact Analysis**: Implementing a Linear Regression model to predict market demand (`owned` proxy) based on game features.
5.  **Business Interpretation**: Translating data findings into actionable business and financial insights.

## Prerequisites

- Python 3.x
- [Jupyter Notebook](https://jupyter.org/install) or [JupyterLab](https://jupyter.org/install)

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd Business_Group_Project_Segmentation_Analysis
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora  # Required for sentiment analysis
```

### 4. Prepare the Data
Ensure the dataset is located in the `DataSet/archive/` directory. 
If you only see `archive.zip` in `DataSet/`, unzip it:
```bash
unzip DataSet/archive.zip -d DataSet/
```
The expected files are:
- `DataSet/archive/bgg-15m-reviews.csv`
- `DataSet/archive/games_detailed_info.csv`

## Running the Project

Open the Jupyter notebook and run all cells:
```bash
jupyter notebook BoardGame_User_Segmentation_and_Revenue_Analysis.ipynb
```

Additionally, you can regenerate the notebook from the source script if needed:
```bash
python create_notebook.py
```

## Technologies Used
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-Learn**: Machine Learning (K-Means, Linear Regression, Scaling)
- **TextBlob**: Natural Language Processing for sentiment analysis
