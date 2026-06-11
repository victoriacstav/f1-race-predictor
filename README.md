# F1 Podium Predictor 🏎️
 
A machine learning project that predicts the probability of a Formula 1 driver finishing on the podium (top 3), based on historical race data.
 
Built together with a fellow student as part of our Applied and Big Data Business Analytics course. We originally started building a stock market prediction model, drawing on our shared interest in economics and finance, but switched to Formula 1, a sport we're both passionate about, to build a project that stood out from our classmates' work.
 
## What it does
 
The model estimates each driver's probability of a podium finish before a race, using their starting grid position and recent form. Predictions are also ranked within each race, producing a predicted top 3 per Grand Prix.
 
## Data
 
Historical Formula 1 data from the [Formula 1 World Championship dataset on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), covering seasons 1950-2024. The analysis is restricted to 2010-2024 (about 5,400 driver-race observations). Only ~17% of entries are podium finishes, so the dataset is imbalanced. All models account for this.
 
## Approach
 
- **Data preparation:** merged race results with driver, constructor, circuit and race metadata using pandas; cleaned missing values (`\N` entries), flagged DNFs, and imputed missing grid positions and lap speeds with medians
- **Feature engineering:** rolling form features computed with `shift(1)` so each prediction only uses information available *before* the race (no data leakage): driver and constructor average finish over the last 5 races, plus grid position indicators (front row, top 5, top 10 start)
- **Train/test split:** time-based rather than random. Trained on 2010-2020 and tested on 2021-2024 to simulate predicting genuinely unseen future seasons
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting and XGBoost, all configured to handle class imbalance (class weights / sample weights / `scale_pos_weight`)
- **Evaluation:** accuracy, precision, recall, F1 and ROC-AUC on the test set, plus confusion matrix, ROC curve, feature importance, and 5-fold TimeSeriesSplit cross-validation
## Results
 
All four models reached a ROC-AUC above 0.91 on unseen seasons (2021-2024):
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.56 | 0.89 | 0.68 | **0.94** |
| Random Forest | 0.89 | 0.68 | 0.72 | **0.70** | 0.92 |
| Gradient Boosting | 0.85 | 0.54 | 0.82 | 0.65 | 0.91 |
| XGBoost | 0.86 | 0.57 | 0.84 | 0.68 | 0.93 |
 
Random Forest gave the best balance between precision and recall (highest F1), while Logistic Regression achieved the highest ROC-AUC. Feature importance confirmed that starting position dominates: starting in the top 5 was by far the strongest predictor of a podium finish, followed by raw grid position. This is consistent with how rarely drivers fight their way to the podium from far back.
 
Cross-validation with TimeSeriesSplit showed stable performance across folds (ROC-AUC around 0.92-0.94), suggesting the results are not driven by a lucky split.
 
## How to run
 
1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
 
2. Download the dataset from Kaggle (link above) and place the CSV files in the same folder as the notebook
3. Open `Formula1.ipynb` and run all cells from top to bottom
## Future work
 
- Add qualifying session times and weather data as features
- Hyperparameter tuning with grid search
- Predict full finishing order instead of podium probability
## Built with
 
Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn
 
