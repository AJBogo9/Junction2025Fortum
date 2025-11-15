# Junction2025Fortum
Repository for Junction 2025 hackathon. The challenge is Fortum's electricity demand forecasting problem.

# Instructions
1. Add the training data 20251111_JUNCTION_training.xlsx to the data folder.
2. Run files transform.ipynb and baseline_model.ipynb in the mentioned order.

# File structure
.
├── baseline_model.ipynb                        # generates the baselinge model data
├── data                                        # contains all csv files
│   ├── 20251111_JUNCTION_training.xlsx         # original data
│   ├── baseline_12m_forecast.csv               # 12-month forecast generated from baseline_model.ipynb
│   ├── baseline_48h_forecast.csv               # 48-hour forecast generated from baseline_model.ipynb
│   └── merged_hourly_for_azure.csv             # original data converted to csv
├── LICENSE
├── README.md
└── transform.ipynb                             # transforms the provided training data into csv
