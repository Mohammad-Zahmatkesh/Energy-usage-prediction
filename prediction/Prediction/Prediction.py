import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# بارگذاری داده‌ها
data = pd.read_csv('C:/Users/useaf/OneDrive/Desktop/datasets.txt', sep=';', parse_dates=[['Date', 'Time']], na_values='?', low_memory=False)

# تغییر نام ستون با تاریخ و زمان
data.rename(columns={'Date_Time': 'DateTime'}, inplace=True)
data.set_index('DateTime', inplace=True)

# تبدیل داده‌ها به نوع عددی
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
data[numeric_cols] = data[numeric_cols].astype(float)

# پر کردن مقادیر گمشده
data.fillna(data.mean(), inplace=True)

# تقسیم داده‌ها به ویژگی‌ها و هدف
X = data[['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = data['Global_active_power']

# تقسیم‌بندی داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ساخت و آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی بر اساس داده‌های آزمایشی
y_pred = model.predict(X_test)

# ارزیابی مدل
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# چاپ نتایج ارزیابی
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Score R²: {r2}")

# تجسم نتایج
plt.figure(figsize=(14, 7))

y_test_sorted = y_test.sort_index()
y_pred_sorted = pd.Series(y_pred, index=y_test_sorted.index)

plt.plot(y_test_sorted.index, y_test_sorted.values, label="Actual", color='Green')
plt.plot(y_pred_sorted.index, y_pred_sorted.values, label="Prediction", color='grey', linestyle='dashdot')
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")
plt.title("Actual vs Predicted Global Active Power Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

