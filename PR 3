import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ------------------------------------------
# 1. Create Dataset
# ------------------------------------------

data = {
    "Hours":[1,2,3,4,5,6,7,8,9,10],
    "Marks":[35,40,45,50,60,65,70,80,90,95]
}

df = pd.DataFrame(data)

print("Student Dataset")
print(df)

# ------------------------------------------
# 2. Features and Target
# ------------------------------------------

X = df[["Hours"]]
Y = df["Marks"]

print("\nInput Feature")
print(X)

print("\nTarget")
print(Y)

# ------------------------------------------
# 3. Split Dataset
# ------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data")
print(X_train)

print("\nTesting Data")
print(X_test)

# ------------------------------------------
# 4. Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, Y_train)

print("\nModel Trained Successfully")

# ------------------------------------------
# 5. Prediction
# ------------------------------------------

prediction = model.predict(X_test)

print("\nActual Marks")
print(list(Y_test))

print("\nPredicted Marks")
print(prediction)

# ------------------------------------------
# 6. Predict New Student
# ------------------------------------------

hours = [[11]]

new_prediction = model.predict(hours)

print("\nPrediction for 11 Study Hours")
print(new_prediction)

# ------------------------------------------
# 7. Model Parameters
# ------------------------------------------

print("\nIntercept")
print(model.intercept_)

print("\nCoefficient")
print(model.coef_)

# ------------------------------------------
# 8. Evaluation
# ------------------------------------------

mae = mean_absolute_error(Y_test, prediction)
mse = mean_squared_error(Y_test, prediction)
r2 = r2_score(Y_test, prediction)

print("\nMean Absolute Error")
print(mae)

print("\nMean Squared Error")
print(mse)

print("\nR2 Score")
print(r2)

# ------------------------------------------
# 9. Multiple Predictions
# ------------------------------------------

new_hours = [[2],[5],[8],[12]]

predicted_marks = model.predict(new_hours)

print("\nMultiple Predictions")

for h, m in zip(new_hours, predicted_marks):
    print("Study Hours =", h[0],
          "Predicted Marks =", round(m,2))

# ------------------------------------------
# 10. Pass / Fail Prediction
# ------------------------------------------

print("\nPass / Fail")

for mark in predicted_marks:

    if mark >= 35:
        print(round(mark,2), "-> PASS")

    else:
        print(round(mark,2), "-> FAIL")

# ------------------------------------------
# 11. Model Summary
# ------------------------------------------

print("\n----------- MODEL SUMMARY -----------")

print("Algorithm : Linear Regression")
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))
print("Features :", list(X.columns))

print("-------------------------------------")

print("\nMachine Learning Process Completed Successfully.")
