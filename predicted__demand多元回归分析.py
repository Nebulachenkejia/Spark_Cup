import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv("demand_features.csv")
X = df[['borrow_count', 'return_count', 'casual_ratio', 'POI_score', 'holiday_factor']]
y = df['predicted_demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
coefficients = model.coef_
intercept = model.intercept_
formula = f"predicted_demand = {intercept:.4f}"
for i, feature in enumerate(X):
    sign = "+" if coefficients[i] >= 0 else "-"
    formula += f" {sign} {abs(coefficients[i]):.4f}×{feature}"

print('模型系数：', coefficients)
print('模型截距：', intercept)
print('R² 得分：', r2)
print('均方误差：', mse)
print("\n模型公式：")
print(formula)