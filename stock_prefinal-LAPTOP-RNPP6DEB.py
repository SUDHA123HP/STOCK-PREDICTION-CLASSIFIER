
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Download data
df = yf.download("AAPL", start="2022-01-01", end="2023-01-01").copy()

# Step 2: Create 'Tomorrow' column (next day's close)
df["Tomorrow"] = df["Close"].shift(-1)

# Step 3: Drop last row (it will have NaN after shift)
df.dropna(inplace=True)

# ✅ Step 4: Fix shape mismatch using .ravel()
df["Target"] = (df["Tomorrow"].to_numpy().ravel() > df["Close"].to_numpy().ravel()).astype(int)

# Step 5: Select features (X) and label (y)
X = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

# Step 6: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 8: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Plot closing prices
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Close"], label="AAPL Closing Price")
plt.title("AAPL Stock Prices (2022)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
