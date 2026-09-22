# LoanSense AI — Loan Approval Prediction

An interactive Streamlit web app that predicts loan approval outcomes using a
**CatBoost** gradient-boosting classifier (94% test accuracy). Applicants'
demographic, financial, loan, and credit-history details are entered through
a form, and the app returns an instant approval prediction with confidence
gauges and probability charts.

## Features

- Interactive prediction form covering all 13 input features
- Real-time approval probability with a gauge chart and bar chart
- Model insights tab with feature-importance and top-driver visualizations
- Polished, finance-themed UI built with Streamlit + Plotly

## Project files

```
.
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container build definition
├── catboost_loan_approval.cbm    # Trained CatBoost model (add this yourself)
└── README.md
```

> **Note:** `catboost_loan_approval.cbm` is not included here — copy your own
> trained model file into this folder before running or deploying the app.

## Running locally

1. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure `catboost_loan_approval.cbm` is in this same folder.
4. Run the app:
   ```bash
   python -m streamlit run app.py
   ```
5. Streamlit prints a local URL (usually `http://localhost:8501`) — open it
   in your browser.

## Running with Docker

Build the image (from this folder, with the `.cbm` model file present):

```bash
docker build -t loansense-ai .
```

Run the container:

```bash
docker run -p 8501:8501 loansense-ai
```

Open `http://localhost:8501` in your browser. To run this on a server, deploy
the same image there and open the server's IP/domain on port 8501 (put it
behind a reverse proxy such as Nginx or Caddy for HTTPS and a custom domain).

## Deploying so anyone can access it online

### Option A — Streamlit Community Cloud (free, easiest)

1. Push this folder (including the `.cbm` model file) to a GitHub repository.
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. Click **New app**, select the repo, branch, and `app.py` as the entry
   point.
4. Click **Deploy**. You'll get a public link like
   `https://your-app-name.streamlit.app` that anyone can open in any
   browser.
5. Pushing new commits to the branch automatically redeploys the app.

### Option B — Hugging Face Spaces (free, Docker-based)

1. Create a new Space, choosing the **Docker** SDK.
2. Push this folder (including the `Dockerfile` and the `.cbm` model file)
   to the Space's git repository.
3. The Space builds the Docker image automatically and gives you a public
   URL.

### Option C — Render / Railway / your own VPS

Use the included `Dockerfile` to build and run the container on any platform
that supports Docker deployments. Point the platform's port mapping at
`8501`, and set up a domain/HTTPS through the platform's built-in proxy (or
Nginx/Caddy on a VPS).

## Notes on the model input format

`app.py` builds a single-row `pandas.DataFrame` with the exact column names
from the feature table below and passes it directly to
`model.predict_proba()`. If your CatBoost model was trained with different
column names, category label casing, or a different feature order, update
the corresponding `selectbox`/`number_input` values in `app.py` to match.

| Column | Description | Data Type |
|---|---|---|
| person_age | Applicant's age | Float |
| person_gender | Applicant's gender | Categorical |
| person_education | Applicant's highest level of education | Categorical |
| person_income | Applicant's annual income | Float |
| person_emp_exp | Years of employment experience | Integer |
| person_home_ownership | Home ownership status | Categorical |
| loan_amnt | Amount of loan requested | Float |
| loan_intent | Intended purpose of the loan | Categorical |
| loan_int_rate | Interest rate applicable to the loan | Float |
| loan_percent_income | Loan amount as % of annual income | Float |
| cb_person_cred_hist_length | Years of credit history | Float |
| credit_score | Applicant's credit score | Integer |
| previous_loan_defaults_on_file | Indicator of previous loan defaults | Categorical |

## License

Add your preferred license here (e.g. MIT) before publishing the repository.
