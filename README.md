# Board Game User Segmentation & Rating-Driven Revenue Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

This project performs a comprehensive analysis of the **BoardGameGeek (BGG)** dataset, featuring over 15 million reviews. It identifies key user personas through machine learning segmentation and predicts market demand for board games based on their community-rated attributes.

## 🚀 Quick Links
- **Live Demo:** [Business Group Project Dashboard](https://businessgroupprojectusersegmentationanalysis-fkmfgpnvaef4zwkp2.streamlit.app/)
- **GitHub Repository:** [RiteshJadhav283/Business_Group_Project_User_Segmentation_Analysis](https://github.com/RiteshJadhav283/Business_Group_Project_User_Segmentation_Analysis)

---

## 📊 Project Overview

The project leverages a dual-pronged approach to board game market analysis:

1.  **User Segmentation (Clustering):** Analyzing 15M+ reviews to group users into distinct behavioral clusters.
2.  **Market Demand Prediction (Regression):** Using game features like complexity, average rating, and community size to estimate total ownership.

### Key Interactive Features
- **EDA Dashboard:** Real-time visualization of rating distributions and market penetration log-scales.
- **User Persona Deep-Dive:** Interactive comparison of "Quality Seekers," "Power Reviewers," and "Casual Raters."
- **Demand Predictor:** A "What-If" tool for publishers to input game specs and predict market demand (Estimated Owners).

---

## 🤖 Machine Learning Implementation

### 👥 User Segmentation (K-Means)
We use **K-Means Clustering** on three engineered user features:
- `num_ratings`: Total reviews submitted.
- `avg_rating`: Average sentiment (1-10 scale).
- `std_rating`: Rating consistency/volatility.

**Target Personas:**
- **Quality Seekers:** High average ratings, selective reviewers.
- **Power Reviewers:** Extremely high activity, the backbone of community engagement.
- **Casual Raters:** Moderate activity, generally satisfied but less vocal.

### 🔮 Demand Prediction (Linear Regression)
A **Linear Regression** model trained on ~20,000+ games to predict the number of owners (`owned` feature).
- **Features:** Average Rating, Community Complexity (Weight), and Total Voter Count.
- **Normalization:** Target variable is **log-transformed** (`log1p`) during training to address the right-skewed nature of market success.

---

## 💡 Business Impact

| Segment | Revenue Strategy |
|:--- |:--- |
| **Quality Seekers** | Target with premium components and high-end aesthetic expansions. |
| **Power Reviewers** | Crucial for "Early Access" programs and community-driven marketing (influencers). |
| **Casual Raters** | Best target for introductory bundles and mass-market retail positioning. |

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- [Kaggle Account](https://www.kaggle.com/) (to download the source dataset)

### 2. Standard Installation
```bash
# Clone the repository
git clone https://github.com/RiteshJadhav283/Business_Group_Project_User_Segmentation_Analysis.git
cd Business_Group_Project_User_Segmentation_Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 3. Data Preparation
1. Download the [BoardGameGeek Reviews Dataset](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews).
2. Place `bgg-15m-reviews.csv` and `games_detailed_info.csv` into `DataSet/archive/`.
3. **Train Models:**
   ```bash
   python train_and_save_models.py
   ```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 🧰 Technologies Used
- **Frontend:** Streamlit, Matplotlib, Seaborn, Plotly
- **Data Engineering:** Pandas, NumPy, Parquet
- **Machine Learning:** Scikit-learn (K-Means, Linear Regression, StandardScaler)
- **Natural Language Processing:** TextBlob (Sentiment extraction)

---

## 👨‍💻 Author
**Ritesh Jadhav**  
*Business Group Project: User Segmentation & Revenue Analysis*
