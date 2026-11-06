# delivery-delay-intelligence-pipeline
Answer to the problem in delivery
# requirements.txt 
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
joblib==1.4.2
# generate_data.py
"""Synthetic shipment data for the Delay Intelligence capstone. Run once:
    python generate_data.py   (writes data/hubs.csv, data/raw_shipments.csv, data/new_shipments.csv)
Do not edit any constant: every number in the brief was measured on this exact output."""
import os
import numpy as np
import pandas as pd

HUBS = [
    ("HUB01", "Mumbai", "West", 1, 1500),
    ("HUB02", "Delhi", "North", 1, 1600),
    ("HUB03", "Bengaluru", "South", 1, 1450),
    ("HUB04", "Hyderabad", "South", 1, 1350),
    ("HUB05", "Chennai", "South", 1, 1300),
    ("HUB06", "Pune", "West", 1, 1250),
    ("HUB07", "Kolkata", "East", 1, 1200),
    ("HUB08", "Ahmedabad", "West", 1, 1150),
    ("HUB09", "Jaipur", "North", 2, 950),
    ("HUB10", "Lucknow", "North", 2, 900),
    ("HUB11", "Chandigarh", "North", 2, 850),
    ("HUB12", "Indore", "West", 2, 820),
    ("HUB13", "Nagpur", "West", 2, 780),
    ("HUB14", "Coimbatore", "South", 2, 740),
    ("HUB15", "Kochi", "South", 2, 700),
    ("HUB16", "Surat", "West", 2, 650),
    ("HUB17", "Patna", "East", 3, 520),
    ("HUB18", "Ranchi", "East", 3, 480),
    ("HUB19", "Guwahati", "East", 3, 450),
    ("HUB20", "Raipur", "East", 3, 420),
    ("HUB21", "Bhubaneswar", "East", 3, 390),
    ("HUB22", "Dehradun", "North", 3, 350),
    ("HUB23", "Siliguri", "East", 3, 310),
    ("HUB24", "Agartala", "East", 3, 270),
]

DIST_MU = [180, 320, 480]       # mean lane distance (km) for hub tier 1 / 2 / 3
DIST_SD = [50, 90, 150]
STOPS_MU = [55, 70, 85]      # mean stops on the rider route for hub tier 1 / 2 / 3
HUB_EFFECT = [0, 2, 4]       # extra hours for hub tier 1 / 2 / 3

# How a hub really operates (1 = metro-like, 2 = regional, 3 = stretched). Four hubs operate unlike their tier label.
OPERATING_TIER = {"HUB07": 2, "HUB13": 3, "HUB15": 3, "HUB22": 2}
BASE_HOURS = {"Express": 10, "Standard": 28, "Economy": 58}

def simulate(n_per_hub, seed):
    """Clean shipments, n_per_hub rows for each of the 24 hubs, in hub order."""
    np.random.seed(seed)
    hubs = pd.DataFrame(HUBS, columns=["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"])
    n = n_per_hub * len(hubs)
    op_tier = [OPERATING_TIER.get(h, t) for h, t in zip(hubs["hub_id"], hubs["hub_tier"])]
    tier = np.repeat(op_tier, n_per_hub) - 1
    hub_id = np.repeat(hubs["hub_id"].values, n_per_hub)

    distance = np.clip(np.random.normal(np.array(DIST_MU)[tier], np.array(DIST_SD)[tier]), 15, 1500).round(1)
    stops = np.clip(np.random.normal(np.array(STOPS_MU)[tier], 10), 20, 140).round().astype(int)
    weight = np.clip(np.random.lognormal(0.6, 0.7, n), 0.1, 40).round(2)
    value = np.random.lognormal(7.0, 0.8, n).round().astype(int)
    service = np.random.choice(["Express", "Standard", "Economy"], n, p=[0.2, 0.55, 0.25])
    pay = np.random.choice(["Prepaid", "COD"], n, p=[0.6, 0.4])
    rain = (np.random.rand(n) < 0.18).astype(int)

    day = np.random.randint(0, 90, n)
    hour = np.random.randint(6, 22, n)
    minute = np.random.randint(0, 60, n)
    ts = pd.Timestamp("2026-07-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 60 + minute, unit="m")

    sale = ((ts >= pd.Timestamp("2026-09-15")) & (ts < pd.Timestamp("2026-09-29"))).astype(int)
    weekend = (ts.dayofweek >= 5).astype(int)

    hours = (
        np.array([BASE_HOURS[s] for s in service])
        + 0.03 * distance
        + 0.10 * stops
        + np.array(HUB_EFFECT)[tier]
        + 0.12 * weight
        + 4 * rain
        + 7 * sale
        + 3 * (pay == "COD")
        + 2 * weekend
        + 3 * (hour >= 18)
        + 12 * sale * (service == "Express")
        + 8 * (distance > 400) * rain
        + np.random.normal(0, 4, n)
    ).round(2)

    return pd.DataFrame({
        "hub_id": hub_id,
        "service_type": service,
        "payment_mode": pay,
        "parcel_weight_kg": weight,
        "distance_km": distance,
        "declared_value": value,
        "n_stops": stops,
        "rain_flag": rain,
        "sale_event_flag": sale,
        "dispatch_time": ts,
        "label_print_batch": np.random.randint(1000, 9999, n),
        "scanner_build_no": np.random.randint(1, 40, n),
        "tape_roll_no": np.random.randint(1, 500, n),
        "sticker_serial_mod": np.random.uniform(0, 1, n).round(3),
        "shelf_slot_no": np.random.randint(1, 300, n),
        "delivery_hours": hours,
    })

def make_messy(df, seed, n_missing, n_case, n_sentinel, n_dupes, id_start):
    """Turn a clean table into a raw export: text columns, gaps, duplicates, a sentinel."""
    np.random.seed(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    df.insert(0, "shipment_id", ["SHP" + str(id_start + i) for i in range(n)])
    df["dispatch_time"] = df["dispatch_time"].dt.strftime("%Y-%m-%d %H:%M")
    df["declared_value"] = df["declared_value"].map(lambda v: "\u20b9" + format(v, ","))

    for col, k in [("distance_km", n_missing[0]), ("parcel_weight_kg", n_missing[1]), ("declared_value", n_missing[2]), ("payment_mode", n_missing[3])]:
        df.loc[np.random.choice(n, k, replace=False), col] = np.nan

    pick = np.random.choice(n, n_case, replace=False)
    shapes = np.random.randint(0, 3, n_case)
    for i, s in zip(pick, shapes):
        v = df.at[i, "service_type"]
        df.at[i, "service_type"] = [v.upper(), v.lower(), "  " + v + " "][s]

    if n_sentinel:
        df.loc[np.random.choice(n, n_sentinel, replace=False), "parcel_weight_kg"] = 999.0

    if n_dupes:
        df = pd.concat([df, df.iloc[np.random.choice(n, n_dupes, replace=False)]], ignore_index=True)

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    hubs = pd.DataFrame(HUBS, columns=["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"])
    hubs.to_csv("data/hubs.csv", index=False)

    raw = make_messy(simulate(50, 2604), seed=7, n_missing=(36, 24, 12, 18), n_case=120, n_sentinel=5, n_dupes=24, id_start=10001)
    raw.to_csv("data/raw_shipments.csv", index=False)

    new = make_messy(simulate(2, 99), seed=8, n_missing=(3, 2, 1, 2), n_case=6, n_sentinel=0, n_dupes=0, id_start=90001)
    new = new.drop(columns=["delivery_hours"])
    new.to_csv("data/new_shipments.csv", index=False)

    print(len(hubs), len(raw), len(new))
    import numpy as np
import pandas as pd

df=pd.read_csv("/content/data/hubs")
df
|index|hub\_id|hub\_city|region|hub\_tier|hub\_daily\_capacity|
|---|---|---|---|---|---|
|0|HUB01|Mumbai|West|1|1500|
|1|HUB02|Delhi|North|1|1600|
|2|HUB03|Bengaluru|South|1|1450|
|3|HUB04|Hyderabad|South|1|1350|
|4|HUB05|Chennai|South|1|1300|
|5|HUB06|Pune|West|1|1250|
|6|HUB07|Kolkata|East|1|1200|
|7|HUB08|Ahmedabad|West|1|1150|
|8|HUB09|Jaipur|North|2|950|
|9|HUB10|Lucknow|North|2|900|
|10|HUB11|Chandigarh|North|2|850|
|11|HUB12|Indore|West|2|820|
|12|HUB13|Nagpur|West|2|780|
|13|HUB14|Coimbatore|South|2|740|
|14|HUB15|Kochi|South|2|700|
|15|HUB16|Surat|West|2|650|
|16|HUB17|Patna|East|3|520|
|17|HUB18|Ranchi|East|3|480|
|18|HUB19|Guwahati|East|3|450|
|19|HUB20|Raipur|East|3|420|
|20|HUB21|Bhubaneswar|East|3|390|
|21|HUB22|Dehradun|North|3|350|
|22|HUB23|Siliguri|East|3|310|
|23|HUB24|Agartala|East|3|270|
# src/shipment_prep.py
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

def clean_shipments(raw, medians=None):
    df = raw.copy()
    
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates()
    
    # 2. service_type formatting
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. payment_mode missing placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. declared_value conversion
    if df["declared_value"].dtype == object:
        clean_val = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df["declared_value_inr"] = pd.to_numeric(clean_val, errors="coerce")
    else:
        df["declared_value_inr"] = df["declared_value"]
    df = df.drop(columns=["declared_value"], errors="ignore")
    
    # 5. dispatch_time datetime conversion
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    
    # 6. parcel_weight_kg scanner-fault sentinel (> 100)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. Compute or apply medians
    cols_to_fill = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {c: float(df[c].median()) for c in cols_to_fill}
        
    for c in cols_to_fill:
        df[c] = df[c].fillna(medians[c])
        
    return df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df
  # part_a_clean_eda.py(Data Foundation and Delay Evidence)
  import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS, FEATURES

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    raw_rows = len(raw)
    
    # Track statistics before dedup / cleaning for checks file
    # Check duplicate count
    dup_mask = raw.duplicated()
    duplicate_rows_removed = dup_mask.sum()
    
    temp_raw = raw.drop_duplicates()
    rows_after_dedup = len(temp_raw)
    
    service_type_spellings_before = temp_raw["service_type"].nunique()
    
    # Parse cleaned service type for after count
    cleaned_service = temp_raw["service_type"].astype(str).str.strip().str.title()
    service_type_spellings_after = cleaned_service.nunique()
    
    # Before missing counts (ignoring duplicates)
    missing_distance_km_before = temp_raw["distance_km"].isna().sum()
    missing_parcel_weight_before = temp_raw["parcel_weight_kg"].isna().loc[temp_raw["parcel_weight_kg"] <= 100].sum()
    sentinel_weight_rows = (temp_raw["parcel_weight_kg"] > 100).sum()
    
    # clean declared value to check missing before fill
    dec_val_clean = temp_raw["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
    missing_declared_value_before = pd.to_numeric(dec_val_clean, errors="coerce").isna().sum()
    missing_payment_mode_before = temp_raw["payment_mode"].isna().sum()
    
    # Run pipeline clean & build
    cleaned_df, medians = clean_shipments(raw)
    modelling_df = build_features(cleaned_df, hubs)
    
    rows_after_hub_merge = len(modelling_df)
    
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)
    
    # Save modelling table
    modelling_df.to_csv("data/modelling_table.csv", index=False)
    
    nulls_in_features_and_target = modelling_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    
    delayed_count = modelling_df["is_delayed"].sum()
    delay_rate = delayed_count / rows_after_hub_merge
    
    declared_value_mean_inr = modelling_df["declared_value_inr"].mean()
    declared_value_median_inr = modelling_df["declared_value_inr"].median()
    log_declared_value_mean = modelling_df["log_declared_value"].mean()
    log_declared_value_median = modelling_df["log_declared_value"].median()
    
    hours_q1 = modelling_df["delivery_hours"].quantile(0.25)
    hours_q3 = modelling_df["delivery_hours"].quantile(0.75)
    iqr = hours_q3 - hours_q1
    hours_lower_bound = hours_q1 - 1.5 * iqr
    hours_upper_bound = hours_q3 + 1.5 * iqr
    hours_outlier_count = ((modelling_df["delivery_hours"] < hours_lower_bound) | (modelling_df["delivery_hours"] > hours_upper_bound)).sum()
    
    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("declared_value_mean_inr", round(declared_value_mean_inr, 4)),
        ("declared_value_median_inr", round(declared_value_median_inr, 4)),
        ("log_declared_value_mean", round(log_declared_value_mean, 4)),
        ("log_declared_value_median", round(log_declared_value_median, 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", round(delay_rate, 4)),
        ("hours_q1 / hours_q3", f"{round(hours_q1, 2)} / {round(hours_q3, 2)}"),
        ("hours_lower_bound / hours_upper_bound", f"{round(hours_lower_bound, 2)} / {round(hours_upper_bound, 2)}"),
        ("hours_outlier_count", hours_outlier_count)
    ]
    pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay by group table
    groups = []
    # 1. service_type
    for st, grp in modelling_df.groupby("service_type"):
        groups.append(("service_type", st, len(grp), int(grp["is_delayed"].sum()), round(grp["is_delayed"].mean(), 4), round(grp["delivery_hours"].mean(), 4)))
    # 2. hub_tier
    for ht, grp in modelling_df.groupby("hub_tier"):
        groups.append(("hub_tier", str(ht), len(grp), int(grp["is_delayed"].sum()), round(grp["is_delayed"].mean(), 4), round(grp["delivery_hours"].mean(), 4)))
    # 3. distance_band
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_df["distance_band"] = pd.cut(modelling_df["distance_km"], bins=bins, labels=labels, right=True, include_lowest=True)
    for db, grp in modelling_df.groupby("distance_band", observed=False):
        groups.append(("distance_band", str(db), len(grp), int(grp["is_delayed"].sum()), round(grp["is_delayed"].mean(), 4), round(grp["delivery_hours"].mean(), 4)))
        
    group_df = pd.DataFrame(groups, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
    group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # Correlations
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_matrix = modelling_df[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True)

if __name__ == "__main__":
    main()
# part_b_regression.py(Delivery-Hours Regression)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

def main():
    os.makedirs("outputs", os.path.exists("outputs"))
    df = pd.read_csv("data/modelling_table.csv")
    
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Split sizes 840 / 180 / 180
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())
    
    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean)
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    
    # Metrics helper
    def evaluate_model(model, X_v, y_v):
        preds = model.predict(X_v)
        mae = mean_absolute_error(y_v, preds)
        mse = mean_squared_error(y_v, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_v, preds)
        return mae, mse, rmse, r2

    metrics_rows = []
    
    # 3. Baseline (mean of y_train)
    baseline_pred = np.full_like(y_val, y_train.mean())
    b_mae = mean_absolute_error(y_val, baseline_pred)
    b_mse = mean_squared_error(y_val, baseline_pred)
    b_rmse = np.sqrt(b_mse)
    b_r2 = r2_score(y_val, baseline_pred)
    metrics_rows.append({"model": "mean_baseline", "split": "val", "MAE": round(b_mae, 4), "MSE": round(b_mse, 4), "RMSE": round(b_rmse, 4), "R2": round(b_r2, 4)})
    
    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_results = []
    best_dt = None
    best_dt_depth = None
    min_val_rmse = float('inf')
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        t_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        
        d_label = str(d) if d is not None else "unlimited"
        depth_results.append({"max_depth": d_label, "train_rmse": round(t_rmse, 4), "val_rmse": round(v_rmse, 4)})
        
        if v_rmse < min_val_rmse:
            min_val_rmse = v_rmse
            best_dt = dt
            best_dt_depth = d_label
            
    pd.DataFrame(depth_results).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # 5. Four candidate models
    # Linear Regression (with scaled inputs)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_mae, lr_mse, lr_rmse, lr_r2 = evaluate_model(lr, X_val_scaled, y_val)
    metrics_rows.append({"model": "linear_regression", "split": "val", "MAE": round(lr_mae, 4), "MSE": round(lr_mse, 4), "RMSE": round(lr_rmse, 4), "R2": round(lr_r2, 4)})
    
    # Decision Tree
    dt_mae, dt_mse, dt_rmse, dt_r2 = evaluate_model(best_dt, X_val, y_val)
    metrics_rows.append({"model": "decision_tree", "split": "val", "MAE": round(dt_mae, 4), "MSE": round(dt_mse, 4), "RMSE": round(dt_rmse, 4), "R2": round(dt_r2, 4)})
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_model(rf, X_val, y_val)
    metrics_rows.append({"model": "random_forest", "split": "val", "MAE": round(rf_mae, 4), "MSE": round(rf_mse, 4), "RMSE": round(rf_rmse, 4), "R2": round(rf_r2, 4)})
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_mae, gb_mse, gb_rmse, gb_r2 = evaluate_model(gb, X_val, y_val)
    metrics_rows.append({"model": "gradient_boosting", "split": "val", "MAE": round(gb_mae, 4), "MSE": round(gb_mse, 4), "RMSE": round(gb_rmse, 4), "R2": round(gb_r2, 4)})
    
    # 6. Learning-rate experiment
    lr_lrs = [0.01, 0.1, 1.0]
    lr_results = []
    for lr_val in lr_lrs:
        gbt = GradientBoostingRegressor(n_estimators=100, learning_rate=lr_val, random_state=42)
        gbt.fit(X_train, y_train)
        v_rmse = np.sqrt(mean_squared_error(y_val, gbt.predict(X_val)))
        v_r2 = r2_score(y_val, gbt.predict(X_val))
        lr_results.append({"learning_rate": lr_val, "val_rmse": round(v_rmse, 4), "val_r2": round(v_r2, 4)})
    pd.DataFrame(lr_results).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 7. Cross-validation comparison & final model selection
    # Determine top 2 models based on validation RMSE
    candidates_eval = [
        ("linear_regression", lr_rmse, lr, X_train_scaled, X_test_scaled),
        ("gradient_boosting", gb_rmse, gb, X_train, X_test),
        ("random_forest", rf_rmse, rf, X_train, X_test),
        ("decision_tree", dt_rmse, best_dt, X_train, X_test)
    ]
    candidates_eval.sort(key=lambda x: x[1])
    top2 = candidates_eval[:2]
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    cv_rows = []
    for name, _, model_obj, _, _ in top2:
        if name == "linear_regression":
            scaler_tv = StandardScaler()
            X_tv_scaled = scaler_tv.fit_transform(X_train_val)
            scores = cross_val_score(LinearRegression(), X_tv_scaled, y_train_val, cv=cv, scoring="r2")
        else:
            scores = cross_val_score(model_obj.__class__(**model_obj.get_params()), X_train_val, y_train_val, cv=cv, scoring="r2")
        cv_rows.append({"model": name, "mean_r2": round(float(scores.mean()), 4), "std_r2": round(float(scores.std()), 4)})
        
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)
    
    # Final model: best val rmse model evaluated once on test set
    best_name, _, best_model, best_X_tr, best_X_te = candidates_eval[0]
    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(best_model, best_X_te, y_test)
    metrics_rows.append({"model": best_name, "split": "test", "MAE": round(test_mae, 4), "MSE": round(test_mse, 4), "RMSE": round(test_rmse, 4), "R2": round(test_r2, 4)})
    
    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)

if __name__ == "__main__":
    main()
# part_c_classification.py(Delay Classifier and Alert Threshold)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def get_cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return int(TN), int(FP), int(FN), int(TP), round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)

def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y)
    
    # 2. Blind baseline and classifiers
    models_rows = []
    
    # Blind baseline (all 0)
    y_pred_blind = np.zeros_like(y_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_pred_blind)
    models_rows.append({"model": "blind_all_on_time", "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": 0.0, "recall": rec, "f1": f1})
    
    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_pred_lr)
    models_rows.append({"model": "logistic_regression", "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    
    # Decision tree depth choice (2 to 10 by highest val F1)
    best_dt = None
    best_dt_d = 2
    best_dt_f1 = -1
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        f1_val = f1_score(y_val, dt.predict(X_val), zero_division=0)
        if f1_val > best_dt_f1:
            best_dt_f1 = f1_val
            best_dt = dt
            best_dt_d = d
            
    y_pred_dt = best_dt.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_pred_dt)
    models_rows.append({"model": f"decision_tree_depth_{best_dt_d}", "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    
    # Random forest default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    y_pred_rf = rf_def.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_pred_rf)
    models_rows.append({"model": "random_forest_default", "TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    
    pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Feature signals & SelectKBest evaluation
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    importances = rf_all.feature_importances_
    
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    
    feat_score_rows = []
    for feat, f_sc, imp in zip(FEATURES, f_scores, importances):
        feat_score_rows.append({"feature": feat, "f_score": round(float(f_sc), 4), "rf_importance": round(float(imp), 4), "is_noise": feat in NOISE_COLUMNS})
        
    feat_score_df = pd.DataFrame(feat_score_rows)
    feat_score_df = feat_score_df.sort_values(by="f_score", ascending=False).reset_index(drop=True)
    feat_score_df["f_rank"] = range(1, len(feat_score_df) + 1)
    feat_score_df = feat_score_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feat_score_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    selectk_rows = []
    for k_val in [5, 7, 10, 12, 17]:
        pipe_k = Pipeline([("selector", SelectKBest(f_classif, k=k_val)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        pipe_k.fit(X_train, y_train)
        f1_k = f1_score(y_val, pipe_k.predict(X_val), zero_division=0)
        selectk_rows.append({"k": k_val, "val_f1": round(float(f1_k), 4)})
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # 4. Tuning with k=7
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_k7 = selector_7.fit_transform(X_train, y_train)
    X_val_k7 = selector_7.transform(X_val)
    
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(rf_base, param_grid, cv=cv_kfold)
    grid_search.fit(X_train_k7, y_train)
    
    rand_search = RandomizedSearchCV(rf_base, param_grid, n_iter=9, cv=cv_kfold, random_state=42)
    rand_search.fit(X_train_k7, y_train)
    
    # Untuned forest with same 7 features
    rf_untuned = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_untuned.fit(X_train_k7, y_train)
    val_f1_untuned = f1_score(y_val, rf_untuned.predict(X_val_k7), zero_division=0)
    val_f1_grid = f1_score(y_val, grid_search.predict(X_val_k7), zero_division=0)
    val_f1_rand = f1_score(y_val, rand_search.predict(X_val_k7), zero_division=0)
    
    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(float(grid_search.best_score_), 4),
            "val_f1": round(float(val_f1_grid), 4)
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rand_search.best_params_),
            "best_cv_accuracy": round(float(rand_search.best_score_), 4),
            "val_f1": round(float(val_f1_rand), 4)
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "n_estimators=100, max_depth=None, min_samples_leaf=1",
            "best_cv_accuracy": np.nan,
            "val_f1": round(float(val_f1_untuned), 4)
        }
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Choose final config based on validation F1 priority: default, grid, randomized
    best_row_name = "default_forest_same_7_features"
    best_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    best_val_f1 = val_f1_untuned
    
    if val_f1_grid > best_val_f1:
        best_val_f1 = val_f1_grid
        best_row_name = "grid_search"
        best_params = grid_search.best_params_
    if val_f1_rand > best_val_f1:
        best_val_f1 = val_f1_rand
        best_row_name = "randomized_search"
        best_params = rand_search.best_params_
        
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **best_params))
    ])
    final_pipeline.fit(X_train, y_train)
    
    # 6. Threshold selection
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_rows = []
    
    chosen_threshold = 0.1
    max_t_for_recall = -1
    
    for t in thresholds:
        y_pred_t = (val_probs >= t).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_pred_t)
        thresh_rows.append({
            "threshold": t, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        if rec >= 0.80 and t > max_t_for_recall:
            max_t_for_recall = t
            chosen_threshold = t
            
    # If none reaches 0.80, default rule says 0.1
    if max_t_for_recall == -1:
        # Check if any threshold meets recall >= 0.80
        valid_rec = [r for r in thresh_rows if r["recall"] >= 0.80]
        if valid_rec:
            chosen_threshold = max(valid_rec, key=lambda x: x["threshold"])["threshold"]
        else:
            chosen_threshold = 0.1
            
    pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # 7. Test evaluation with chosen threshold
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(X_test_labels := y_test, y_pred_test)
    
    pd.DataFrame([{
        "threshold": chosen_threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }]).to_csv("outputs/part_c_final_test.csv", index=False)
    
    max_depth_val = best_params.get("max_depth", "unlimited")
    if max_depth_val is None:
        max_depth_val = "unlimited"
        
    pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": best_params.get("n_estimators", 100),
        "max_depth": max_depth_val,
        "min_samples_leaf": best_params.get("min_samples_leaf", 1),
        "threshold": chosen_threshold,
        "chosen_from": best_row_name
    }]).to_csv("outputs/part_c_final_config.csv", index=False)

if __name__ == "__main__":
    main()
  # part_d_segments.py(Hub Segments, Packaging and Operations Playbook)
  import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)
    
    # 1. Hub Profile
    hub_grp = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index()
    
    # Sort by ascending hub_id explicitly
    hub_grp = hub_grp.sort_values(by="hub_id").reset_index(drop=True)
    hub_profile_out = hub_grp.copy()
    hub_profile_out["mean_delivery_hours"] = hub_profile_out["mean_delivery_hours"].round(4)
    hub_profile_out["delay_rate"] = hub_profile_out["delay_rate"].round(4)
    hub_profile_out["mean_distance_km"] = hub_profile_out["mean_distance_km"].round(4)
    hub_profile_out["mean_n_stops"] = hub_profile_out["mean_n_stops"].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)
    
    # 2. K-Means clustering (K=3)
    feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_grp[feature_cols])
    
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_3 = kmeans_3.fit_predict(scaled_matrix)
    
    # 3. Separation evidence for K in 2, 3, 4, 5
    def calc_separation(matrix, labels):
        n = len(matrix)
        same_dist = []
        diff_dist = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(matrix[i] - matrix[j])
                if labels[i] == labels[j]:
                    same_dist.append(dist)
                else:
                    diff_dist.append(dist)
        mean_diff = np.mean(diff_dist) if diff_dist else 0.0
        mean_same = np.mean(same_dist) if same_dist else 1.0
        return mean_diff / mean_same if mean_same > 0 else 0.0

    sep_rows = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(scaled_matrix)
        ratio = calc_separation(scaled_matrix, lbls)
        
        # cluster sizes descending joined by hyphen
        unique_lbls, counts = np.unique(lbls, return_counts=True)
        sorted_counts = sorted(counts, reverse=True)
        sizes_str = "-".join(map(str, sorted_counts))
        sep_rows.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": sizes_str
        })
    pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)
    
    # 4. Segment names and tier check
    hub_grp["cluster"] = labels_3
    
    # Order clusters by mean delay_rate: lowest -> Standard, middle -> Watch, highest -> Escalate
    cluster_delay = hub_grp.groupby("cluster")["delay_rate"].mean().sort_values()
    sorted_clusters = cluster_delay.index.tolist()
    
    name_map = {
        sorted_clusters[0]: "Standard",
        sorted_clusters[1]: "Watch",
        sorted_clusters[2]: "Escalate"
    }
    hub_grp["segment"] = hub_grp["cluster"].map(name_map)
    
    hubs_meta = pd.read_csv("data/hubs.csv")
    merged_hub_info = hub_grp.merge(hubs_meta, on="hub_id")
    
    hub_segments_out = merged_hub_info[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments_out.to_csv("outputs/hub_segments.csv", index=False)
    
    # Segment profile (mean values of hub-level profile metrics)
    seg_profile = merged_hub_info.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    ).reset_index()
    
    # Re-order rows strictly by Standard, Watch, Escalate
    seg_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
    seg_profile["order"] = seg_profile["segment"].map(seg_order)
    seg_profile = seg_profile.sort_values(by="order").drop(columns=["order"])
    
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        seg_profile[col] = seg_profile[col].round(4)
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    # Tier vs Segment mismatch
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
    merged_hub_info["implied_segment"] = merged_hub_info["hub_tier"].map(tier_implied)
    mismatches = merged_hub_info[merged_hub_info["segment"] != merged_hub_info["implied_segment"]]
    mismatches[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

if __name__ == "__main__":
    main()
 # package_model.py 
 import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, clean_shipments, build_features

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")
    raw = pd.read_csv("data/raw_shipments.csv")
    
    # Recompute clean medians exactly as in Part A
    _, medians = clean_shipments(raw)
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # Stratified split matching Part C
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y)
    
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    threshold = float(config["threshold"])
    k_val = int(config["selector_k"])
    n_est = int(config["n_estimators"])
    max_d = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    min_leaf = int(config["min_samples_leaf"])
    
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_val)),
        ("model", RandomForestClassifier(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_leaf, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train)
    
    # Save assets
    joblib.dump(final_pipeline, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    pd.DataFrame(list(medians.items()), columns=["column", "median"]).to_csv("models/impute_values.csv", index=False)
    
    # Reload check
    loaded_model = joblib.load("models/delay_model.pkl")
    in_mem_probs = final_pipeline.predict_proba(X_test)[:, 1]
    loaded_probs = loaded_model.predict_proba(X_test)[:, 1]
    max_diff = float(abs(in_mem_probs - loaded_probs).max())
    
    check_rows = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", max_diff),
        ("threshold", threshold)
    ]
    pd.DataFrame(check_rows, columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

if __name__ == "__main__":
    main()
predict_delay.py
import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def main():
    os.makedirs("outputs", exist_ok=True)
    
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Load models & metadata
    model = joblib.load("models/delay_model.pkl")
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])
    
    feat_order_df = pd.read_csv("models/feature_order.csv")
    feature_list = feat_order_df["feature"].tolist()
    
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))
    
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    
    # Clean and build features using shared src functions
    clean_new, _ = clean_shipments(new_raw, medians=medians)
    features_new = build_features(clean_new, hubs)
    
    X_new = features_new[feature_list]
    probs = model.predict_proba(X_new)[:, 1]
    
    features_new["delay_probability"] = np.round(probs, 4)
    features_new["flagged"] = (features_new["delay_probability"] >= threshold).astype(int)
    
    # Join hub segments
    merged = features_new.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
    
    # Action assignment rule:
    # Escalate to hub manager: flagged == 1 AND segment == "Escalate"
    # Proactive customer call: flagged == 1 AND segment != "Escalate"
    # No action: flagged == 0
    def determine_action(row):
        if row["flagged"] == 0:
            return "No action"
        elif row["segment"] == "Escalate":
            return "Escalate to hub manager"
        else:
            return "Proactive customer call"

    merged["action"] = merged.apply(determine_action, axis=1)
    
    out_df = merged[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    out_df.to_csv("outputs/new_shipment_actions.csv", index=False)

if __name__ == "__main__":
    main()
    README.md Template Structure
My README.md must be placed in the repository root and structured with the following sections, fully populated with the exact metric figures from your generated CSV files:
# Delhivery Last-Mile Delay Intelligence Pipeline

## How to Run
To run the entire pipeline cleanly from scratch on a new clone, execute the following shell commands in sequence:
```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
(Note: models/delay_model.pkl is regenerated via package_model.py. The committed pickle loads reliably only on the scikit-learn version pinned in requirements.txt).
# Data findings
1.Service type delay:Express service is delayed most often at a rate of 48.97%.
2.Hub tier progression: Delay rate increases significantly across tiers, rising from 4.00% in Tier 1 hubs to 21.50% in Tier 2 and 44.00% in Tier
3.Distance bands: Delay rates climb sharply with distance, from 2.19% for 0–200 km up to 85.23% for shipments traveling 601+ km.
4.Strongest correlations with delivery hours: promised_hours (0.837), distance_km (0.401), and n_stops (along with further correlations listed in outputs/part_a_correlations.csv).
5.Declared value transformation: declared_value_inr was natural log-transformed to stabilize right-skewness and normalize its heavy-tailed distribution before modeling.
# Model results (regression)
1.Baseline R²: -0.021, indicating that predicting the mean is worse than a flat guess due to variance.
2.Best validation model: Linear regression and gradient boosting both scored 0.931 on validation R².
3.Tree depth diagnostic: Shallow depths (e.g., 1) exhibit high bias, whereas unlimited depths suffer from high variance (large train-val error gap).
4.Test performance: The final model achieved an R² of 0.949 on the test dataset.  
# Model results (classification)
1.Blind baseline accuracy: 0.7667 (predicts all on-time).
2.Feature signal: Noise columns accounted for 22.5% of random forest importance and all fell outside top ranking features.
3.Threshold & metrics: Chosen alert threshold of 0.5 balances precision (0.689) and recall (0.756) under operational cost constraints.
# Operations recommendation
Summary of Escalate hubs, mismatch corrections based on tier_vs_segment_mismatch.csv, action counts for the 48 new shipments, and justification for keeping $K=3$ clusters.
# Production plan
Maps reliability, reproducibility, performance, and monitoring metrics (e.g., weekly delayed share, recall tracking) with a scheduled retraining plan accounting for seasonal surge events like the 15–28 September sale window.
# Limitations
Synthetic data nature, small validation/test delay sample size, and threshold sensitivity to a single validation split.
