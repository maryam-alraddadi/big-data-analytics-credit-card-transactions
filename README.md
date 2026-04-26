# Big Data Analytics - Credit Card Transactions

Minimal Spark-first project scaffold for credit-card transaction analytics with reproducible data preparation output to Parquet.

## Project structure

- `data/raw/` - input CSV files (not committed).
- `data/processed/` - cleaned/standardized Parquet outputs.
- `src/pipeline.py` - Spark ingestion + cleansing + schema standardization pipeline.
- `dashboard/app.py` - minimal Streamlit viewer for processed data.
- `notebooks/` - optional phase notebooks.
- `requirements.txt` - Python dependencies for local/Colab.

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the prep pipeline.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline \
  --input data/raw/creditcard.csv \
  --output data/processed/transactions_parquet
```

Run the dashboard locally:

```bash
streamlit run dashboard/app.py
```

## Google Colab setup

In a Colab notebook:

```python
!git clone <your-repo-url>
%cd big-data-analytics-credit-card-transactions
!pip install -r requirements.txt
!python -m src.pipeline --input data/raw/creditcard.csv --output data/processed/transactions_parquet
```

If data is in Google Drive, mount Drive and pass absolute input/output paths:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example:

```python
!python -m src.pipeline \
  --input "/content/drive/MyDrive/creditcard.csv" \
  --output "/content/drive/MyDrive/transactions_parquet"
```

## Data preparation behavior

The pipeline currently performs:

- CSV ingestion with robust header handling and normalized column names.
- Schema standardization for common credit-card datasets.
- Type casting (`amount`, `is_fraud`, timestamps).
- Null/invalid filtering and duplicate removal.
- Stable `transaction_id` generation when missing.
- Parquet write (`overwrite` mode by default).

## Notes

- Keep large datasets out of git; place them in `data/raw/`.
- Processed Parquet is the reusable input for later EDA/modeling phases.
