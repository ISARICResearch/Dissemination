# Federated Analytics Workshop

This project compares the **centralized** and **federated** approaches to the
statistical analyses commonly used in clinical epidemiology. 

- In the **Centralized approach**, individual-level data from every institution is pooled in a single
environment and analyzed together. 
- In the **Federated approach**, the data stays under the control of each institution, which runs the analysis locally and
shares only aggregated results (summary statistics or model parameters). These results are then combined into an overall result without transferring any
patient record.

Each analysis is performed with both approaches, and their
results are compared to evaluate whether they match and where they diverge.
As an introductory example, both approaches are evaluated on a dummy dataset of
dengue patient records from 5 health institutions (see [Dataset](#dataset)).

## Notebooks

The project consists of five notebooks, located in the `notebooks/` folder.
Each one covers a specific type of analysis:

| Notebook | Analysis |
|---|---|
| `01_eda.ipynb` | **Exploratory data analysis (EDA):** dataset structure, variable definitions, data quality (missing values), and distribution of key variables overall and by institution. |
| `02_descriptive_analysis.ipynb` | **Descriptive statistics:** mean, standard deviation, median and quartiles for continuous variables, and frequencies for categorical variables. |
| `03_odds_ratios.ipynb` | **Risk factor analysis:** univariate logistic regression for mortality, reporting odds ratios with 95% confidence intervals and p-values. |
| `04_survival_analysis.ipynb` | **Survival analysis:** Kaplan-Meier survival probabilities by age group and a univariate Cox proportional hazards model (hazard ratio), over a 30-day follow-up. |
| `05_prediction.ipynb` | **Clinical prediction model:** multivariable logistic regression with L2 (Ridge) regularization to predict mortality, tuned by cross-validation and evaluated on a held-out test set (AUC, recall, precision, F1 score and accuracy). |

## Project structure

```
├── data/
│   ├── dummy_data.parquet            # dummy dataset
│   └── dummy_data_dictionary.json    # data dictionary
├── analysis_lib/                     # functions used by the notebooks
│   ├── preprocessing.py
│   ├── eda.py
│   ├── descriptive_analysis.py
│   ├── odds_ratios.py
│   ├── survival_analysis.py
│   └── prediction.py
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_descriptive_analysis.ipynb
    ├── 03_odds_ratios.ipynb
    ├── 04_survival_analysis.ipynb
    └── 05_prediction.ipynb
```

### `analysis_lib`

The functions each notebook uses (tables, plots, calculations) are in the
`analysis_lib/` folder. The module most closely related to each notebook has
the same name as that notebook (e.g. `odds_ratios.py` for
`03_odds_ratios.ipynb`). The `preprocessing.py` module holds general data
preparation functions (dropping, imputing and encoding columns by data
dictionary category) shared by most notebooks.

## Dataset

The `data/` folder contains a **dummy** dataset of 10,284 dengue patient records
from 5 health institutions, together with its data dictionary. Its 41 variables
are grouped into the following categories:

- **Identification**: Record identifier.
- **Institution**: Health institution of each record.
- **Date**: Consultation, Symptom onset, Hospitalization and Death dates.
- **Demographic**: Age and Sex.
- **Symptom**: 22 Symptoms and clinical signs (e.g. fever, vomiting, shock,
  severe bleeding).
- **Exam**: Whether a tissue sample was taken, and from which organ.
- **Medical Care**: Final classification, hospitalization and outcome (alive
  or dead).

The data dictionary describes each variable: its description, category, data
type and possible values.

## Requirements

### Google Colab (recommended)

Open a notebook with its **Open in Colab** badge. Only a Google account is
needed. The notebook's setup section clones this repository to get the data and
`analysis_lib`, and installs any package that Colab doesn't provide.

### Local

- Python 3.12.
- Packages: `pandas`, `numpy`, `pyarrow`, `matplotlib`, `scipy`, `statsmodels`,
  `scikit-learn` (version 1.6.1, to reproduce `05_prediction.ipynb`'s results),
  and `jupyter` or another notebook environment.
- Clone the repository and run the notebooks from the `notebooks/` folder,
  since they read the data and `analysis_lib` from its parent folder.