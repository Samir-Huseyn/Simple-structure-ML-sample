import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("listings.csv")
print(df.info())
print(df.describe())
print(df.shape)

df=df.drop(columns=["id", "host_name", "license", "last_review", "neighbourhood_group","name", "neighbourhood"])
print(df.isnull().sum())
df["price"] = df["price"].fillna(df["price"].median())
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df = pd.get_dummies(df, drop_first= True)


X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("mae:", mae)
print("r2 score:", r2)