![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Stars](https://img.shields.io/github/stars/ahtashamd321/mental-health-social-media)
![Issues](https://img.shields.io/github/issues/ahtashamd321/mental-health-social-media)




# 🧠 Mental Health & Social Media Balance Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1.3-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **A comprehensive data science portfolio project analyzing the relationship between social media usage patterns and mental health indicators across 500 users.**

[🔗 Live Dashboard](https://mental-health-social-media.streamlit.app/) | [💼 LinkedIn Profile](https://linkedin.com/in/ahtasham-anjum) | [📊 Portfolio](https://datascienceportfol.io/ahtashamd321)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

This project analyzes mental health indicators and social media usage patterns to uncover insights about digital wellbeing. Using machine learning, statistical analysis, and interactive visualizations, we identify key factors influencing happiness and provide actionable recommendations.

### 🎓 Skills Demonstrated

- **Data Analysis**: Exploratory Data Analysis (EDA), Statistical Testing, Correlation Analysis
- **Machine Learning**: Regression Models (Linear, Random Forest, Gradient Boosting)
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Web Development**: Streamlit Dashboard
- **Statistical Methods**: ANOVA, T-tests, Pearson Correlation, Feature Importance
- **Business Intelligence**: Segmentation, Recommendation Systems, A/B Test Design

---

## 🔍 Key Findings

### Top Insights

1. **Sleep Quality is King** 👑
   - Strongest predictor of happiness (correlation: 0.68)
   - More important than screen time or exercise

2. **Screen Time Sweet Spot** 📱
   - Optimal: 3-5 hours/day → Happiness: 9.8/10
   - High usage (9+ hours) → Happiness: 5.1/10
   - U-shaped relationship detected

3. **Platform Matters** 🎯
   - LinkedIn & Twitter users: 8.9/10 happiness
   - TikTok users: Higher stress (7.2/10) despite popularity
   - Professional platforms correlate with better wellbeing

4. **Digital Detox Works** 🔄
   - 3-5 days/month without social media boosts happiness by 0.5 points
   - Positive correlation: 0.52

5. **The Optimal Formula** ⭐
   ```
   3-5 hrs screen time + 7+ sleep quality + 3+ breaks/month + 3+ exercise/week
   = 9.2/10 Happiness Score
   ```

### Statistical Validation

- **Sample Size**: 500 users
- **ML Model Accuracy**: 87% (R² = 0.76)
- **Cross-Validation Score**: 0.74 ± 0.03
- **Top Feature**: Sleep Quality (28% importance)

---

## 📊 Dataset

### Overview

- **Source**: Mental Health and Social Media Balance Dataset
- **Size**: 500 users
- **Features**: 9 variables
- **Target**: Happiness Index (1-10 scale)

### Variables

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| User_ID | String | Unique identifier | U001-U500 |
| Age | Integer | User age | 16-49 years |
| Gender | Categorical | Male/Female/Other | 3 categories |
| Daily_Screen_Time(hrs) | Float | Average daily usage | 1.0-11.0 hours |
| Sleep_Quality(1-10) | Integer | Self-reported sleep quality | 1-10 |
| Stress_Level(1-10) | Integer | Self-reported stress | 1-10 |
| Days_Without_Social_Media | Integer | Monthly breaks | 0-9 days |
| Exercise_Frequency(week) | Integer | Weekly exercise | 0-7 days |
| Social_Media_Platform | Categorical | Primary platform | 6 platforms |
| Happiness_Index(1-10) | Integer | Overall happiness | 1-10 (Target) |

### Data Quality

- ✅ No missing values
- ✅ No duplicates
- ✅ Balanced gender distribution
- ✅ Wide age range representation

---

## 📁 Project Structure

```
mental-health-social-media/
│
├── data/
│   └── Mental_Health_and_Social_Media_Balance_Dataset.csv
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_statistical_testing.ipynb
│   └── 03_machine_learning.ipynb
│
├── src/
│   ├── statistical_analysis.py
│   ├── visualizations.py
│   └── ml_models.py
│
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── outputs/
│   ├── executive_dashboard.png
│   ├── screen_time_analysis.png
│   ├── feature_importance.png
│   ├── segment_analysis.png
│   └── interactive_dashboard.html
│
└── docs/
    ├── methodology.md
    └── findings_report.pdf
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip package manager
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/ahtashamd321/mental-health-social-media.git
cd mental-health-social-media
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Run Statistical Analysis

```bash
python src/statistical_analysis.py
```

**Output**: Comprehensive statistical report with correlations, ANOVA tests, and insights.

### 2. Generate Visualizations

```bash
python src/visualizations.py
```

**Output**: 5 high-quality PNG files and 1 interactive HTML dashboard.

### 3. Launch Interactive Dashboard

```bash
streamlit run app.py
```

**Access**: Open browser at `http://localhost:8501`

### 4. Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` folder to explore interactive analyses.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

- Descriptive statistics
- Distribution analysis
- Missing value treatment
- Outlier detection

### 2. Statistical Testing

- **Correlation Analysis**: Pearson & Spearman correlations
- **ANOVA**: Platform and age group comparisons
- **T-tests**: Gender-based differences
- **Significance Level**: α = 0.05

### 3. Feature Engineering

- Age groups: 16-20, 21-30, 31-40, 41-50
- Screen time categories: 1-3hrs, 3-5hrs, 5-7hrs, 7-9hrs, 9+hrs
- Wellbeing score: Composite metric
- User segmentation: Thriving, Balanced, At Risk, Critical

### 4. Machine Learning Models

#### Models Evaluated

1. **Linear Regression** (Baseline)
   - R² Score: 0.72
   - MAE: 0.85

2. **Random Forest Regressor** (Best Model) ⭐
   - R² Score: 0.76
   - MAE: 0.73
   - Cross-validation: 0.74 ± 0.03

3. **Gradient Boosting**
   - R² Score: 0.75
   - MAE: 0.76

#### Model Selection Criteria

- Cross-validation performance
- Feature importance interpretability
- Prediction accuracy
- Generalization capability

### 5. Validation

- 80-20 train-test split
- 5-fold cross-validation
- Residual analysis
- Feature importance validation

---

## 📈 Results

### Correlation Matrix

| Variable | Happiness Correlation | P-value |
|----------|----------------------|---------|
| Sleep Quality | +0.68 | < 0.001*** |
| Stress Level | -0.62 | < 0.001*** |
| Screen Time | -0.58 | < 0.001*** |
| Days Without SM | +0.52 | < 0.001*** |
| Exercise Frequency | +0.45 | < 0.001*** |
| Age | +0.08 | 0.142 (ns) |

***: Highly significant (p < 0.001)

### Platform Comparison

| Platform | Avg Happiness | Avg Stress | User Count |
|----------|--------------|------------|------------|
| LinkedIn | 8.9 | 6.2 | 72 |
| Twitter | 8.9 | 6.3 | 89 |
| YouTube | 8.8 | 6.4 | 63 |
| Facebook | 8.7 | 6.1 | 78 |
| Instagram | 8.2 | 7.1 | 100 |
| TikTok | 8.1 | 7.2 | 98 |

### User Segmentation

| Segment | Percentage | Characteristics |
|---------|-----------|-----------------|
| Thriving | 32% | Low screen time, good sleep, low stress |
| Balanced | 41% | Moderate across all metrics |
| At Risk | 21% | High screen time or poor sleep |
| Critical | 6% | Multiple negative indicators |

---

## 🎨 Visualizations

### 1. Executive Dashboard
![Executive Dashboard](outputs/executive_dashboard.png)
*Comprehensive overview with 8 key visualizations*

### 2. Screen Time Analysis
![Screen Time](outputs/screen_time_analysis.png)
*Deep dive into screen time effects*

### 3. Feature Importance
![Features](outputs/feature_importance.png)
*ML model insights*

### 4. Interactive Dashboard
[View Interactive HTML](outputs/interactive_dashboard.html)
*3D scatter plots, dynamic filters, hover interactions*

---

## 🛠️ Technologies Used

### Data Science Stack

- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions

### Machine Learning

- **Scikit-learn**: ML algorithms and preprocessing
- **Random Forest**: Primary predictive model
- **Cross-validation**: Model validation

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web dashboard

### Development Tools

- **Jupyter**: Interactive notebooks
- **Git**: Version control
- **VSCode**: IDE

---

## 🚀 Future Improvements

### Short-term (Next 2 months)

- [ ] Add time-series forecasting
- [ ] Implement clustering algorithms (K-means, DBSCAN)
- [ ] Create A/B test simulator
- [ ] Add confidence intervals to predictions
- [ ] Deploy on Streamlit Cloud/Heroku

### Medium-term (3-6 months)

- [ ] Collect longitudinal data
- [ ] Build recommendation system
- [ ] Add NLP for sentiment analysis
- [ ] Create mobile app version
- [ ] Integrate with wearable device data

### Long-term (6-12 months)

- [ ] Real-time data pipeline
- [ ] Causal inference analysis
- [ ] Deep learning models
- [ ] Multi-language support
- [ ] API for third-party integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Include docstrings

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Ahtasham Anjum**

- 📧 Email: ahtashamd321@gmail.com
- 💼 LinkedIn: [linkedin.com/in/ahtasham-anjum](https://linkedin.com/in/ahtasham-anjum)
- 🐱 GitHub: [@ahtashamd321](https://github.com/ahtashamd321)
- 🌐 Portfolio: [datascienceportfol.io/ahtashamd321](https://datascienceportfol.io/ahtashamd321)

---

## 🙏 Acknowledgments

- Dataset source: Kaggle Mental Health & Social Media Dataset
- Inspiration: Digital Wellbeing Research
- Tools: Streamlit, Plotly, Scikit-learn communities

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/ahtashamd321/mental-health-social-media)
![GitHub forks](https://img.shields.io/github/forks/ahtashamd321/mental-health-social-media)
![GitHub issues](https://img.shields.io/github/issues/ahtashamd321/mental-health-social-media)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ahtashamd321/mental-health-social-media)

---

## 🎓 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{mentalhealth2024,
  author = {Ahtasham Anjum},
  title = {Mental Health & Social Media Balance Analytics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ahtashamd321/mental-health-social-media}
}
```

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ and ☕ by Ahtasham Anjum

[Back to Top](#-mental-health--social-media-balance-analytics)

</div>


