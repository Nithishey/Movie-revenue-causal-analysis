# Movie Revenue Driver Analysis Using Causal Models

What actually drives a movie's box-office revenue? This project goes beyond correlation and uses **causal inference** to estimate the effect of **budget** and **franchise status** on revenue, then presents the results in an interactive **Streamlit** dashboard.

## Dataset
TMDB Box Office data (`train.csv`, 3,000 movies, 23 columns) from the *TMDB Box Office Prediction* competition on Kaggle. The raw file is not included in this repository; download it from Kaggle to re-run the notebook.

## Approach
1. **Data cleaning:** fixed 2-digit-year dates, treated `budget = 0` (27% of rows) as missing, removed a hidden BOM character.
2. **Feature engineering:** parsed JSON-like columns (genres, cast, crew, companies) into counts, flags and genre dummies; added release-timing features and log transforms.
3. **EDA:** revenue skew, budget vs revenue, genre, seasonality, franchise effect.
4. **Causal analysis:**
   - Causal graph (DAG) separating confounders, treatment, outcome and post-treatment variables
   - Backdoor adjustment with OLS (robust standard errors) and inverse probability weighting
   - Refutation tests: placebo treatment, random common cause, data subsets, data-quality sensitivity
   - Cross-check with the DoWhy library

## Key results
| Question | Naive | Adjusted | Note |
|---|---|---|---|
| Budget elasticity (log-log) | 0.78 | **0.70** (95% CI about 0.57 to 0.83) | 10% higher budget goes with about 7% higher revenue |
| Franchise effect (log points) | 1.81 | **1.38** (IPW: 1.35) | roughly a 4x revenue multiple |

Adjusting for popularity and cast size (consequences of budget) lowers the budget estimate to about 0.56, an example of "bad control" bias. The budget estimate ranges from 0.59 to 0.82 depending on how very small revenue values are handled.

## Dashboards
**Streamlit app** (`streamlit_app/app.py`): Overview, Explorer, Causal results and a Budget what-if page, with sidebar filters.

![Streamlit overview](Movie_revenue_causal_analysis/Screenshots/streamlit_overview.png)
![Streamlit causal results](docs/streamlit_causal.png)

**Power BI dashboard** (`powerbi/`): three pages (Overview, Drivers, Causal Results) built on a three-table model, with a budget what-if slider linked to the causal estimate. Open `movie_revenue_driver_dashboard.pbix` in Power BI Desktop, or view the PDF export.

![Power BI overview](docs/powerbi_overview.png)
![Power BI causal results](docs/powerbi_causal.png)

Presentation: `docs/Movie_Revenue_Driver_Analysis.pptx`

## Project structure
```
notebooks/   Colab notebook: cleaning, EDA, causal analysis
streamlit_app/
  app.py             dashboard code
  data/              cleaned data and causal results used by the app
powerbi/     Power BI file (.pbix) and PDF export
docs/        presentation and dashboard screenshots
requirements.txt
```

## Run the dashboard locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

## Limitations
- Unmeasured confounders (star power, marketing spend, director reputation) may bias estimates.
- Movies with unknown budget (27%) are excluded from the budget analysis.
- The linear log-log model estimates an average effect, not the effect for any single film.
- Conclusions depend on the assumptions encoded in the DAG.

## Tools
Python, pandas, NumPy, scikit-learn, NetworkX, Matplotlib, Seaborn, DoWhy, Streamlit, Plotly, Power BI (DAX)
