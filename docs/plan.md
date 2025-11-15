# Features

1. Temporal patterns (AVAILABLE FROM DATA)
    - trends (MA, RSI, etc..) (both)
    - lagged values (12, 24, 48 hours, week a go, monthly, half year) (both)


2. Calendar events (PARTLY AVAILABLE FROM DATA)
    - hour of day, day of week or weekday vs weekend (short)
    - month or season of year (long)

    - holidays/events (is cristmas, etc) (short)
    - workday/weekend count (long)


3. customer metadata (AVAILABLE FROM DATA)
- NOTICE: Group for high level geography -> Better weather indicators (No need for location feature!?)
    - segment
    - product type
    - see all from metadata...


4. weather factors (REQUIRES ADDITIONAL DATA)
- NOTICE: Difference in different regions

    - temperature (current, hourly forecast from weather API, etc) (short)
    - Averaged temperature measurement given past years data (HDD, CDD, etc) (long)

    - optional: humidity/rain/wind (short)
    - daylight (avg over day, week, month, etc) (both)


5. electricity markets (REQUIRES ADDITIONAL DATA)
    - day a head price (short)
        - price change (between days)
        - temporal features
    - optional: Other market features (volume, etc; see market place)


6. Economic factors (REQUIRES ADDITIONAL DATA)
    - optional: see [gpt-features.pdf](gpt-features.pdf) (long)


# Models
- NOTICE: Different models per time horizons and per groups (grouping according region!?)
- NOTICE: Target gain instead of consumption -> less biased!?


