# How to run the model

## Prepare Training data

   Place the original excel here:

   - `data/20251111_JUNCTION_training.xlsx`
   - Run `transform.ipynb` with jupyter notebook vscode extension
   - This creates training data  inside `data/`.

## Testing the model

For testing the short term model, we remove the **last 2 days** from the dataset and keep them as a **reference set**.  
The notebook `validation_baseline.ipynb`:

- Builds the organiser’s baseline model.
- Creates the 2-day reference datasets that are used for testing.
- Allows us to monitor the model predictions for those 2 days and compare them against the reference using our evaluation logic (e.g. via the evaluation notebook/model).

1. **Evaluate the FVA**

   - Run `validation_baseline.ipynb`.
   - This generates the organiser’s baseline forecasts and the 2-day reference sets for testing.
   - Run `evaluation.ipynb` for getting the FVA for the model's predictions

2. **Run short-term model**

   - Go to the `short-term-forecast/` folder.
   - Run the notebooks in the order indicated by their numbering / titles.
   - The final notebook(s) write out our short-term forecast CSV files.

---

# Where the results are stored

- **Baseline model outputs**
  - `data/baseline_48h_forecast.csv` – 48-hour baseline forecast  
  - `data/baseline_12m_forecast.csv` – 12-month baseline forecast  

- **2-day reference test sets**
  - Created by `validation_baseline.ipynb` and saved into `data/`  

- **Our short-term model outputs**
  - Forecast CSV files written by the notebooks in `short-term-forecast/`  

- **Intermediate / processed data**
  - `data/merged_hourly_for_azure.csv` 
  - Any additional helper CSVs created by `transform.ipynb` .
