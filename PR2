import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Create Sample Dataset
# ------------------------------------------

data = {
    "Student": ["Deep", "Rahul", "sarthi", "krish", "Karan", "Amit", "heet"],
    "Maths": [85, 76, 92, 88, 79, 91, 84],
    "Science": [82, 80, 95, 90, 81, 89, 86],
    "English": [78, 74, 90, 87, 75, 88, 80]
}

df = pd.DataFrame(data)

print("Student Dataset")
print(df)

# ------------------------------------------
# 2. Dataset Information
# ------------------------------------------

print("\nShape :", df.shape)
print("\nColumns")
print(df.columns)

print("\nSummary Statistics")
print(df.describe())

# ------------------------------------------
# 3. Line Chart
# ------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(df["Student"], df["Maths"], marker='o')
plt.title("Maths Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

# ------------------------------------------
# 4. Bar Chart
# ------------------------------------------

plt.figure(figsize=(8,5))
plt.bar(df["Student"], df["Science"])
plt.title("Science Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# ------------------------------------------
# 5. Scatter Plot
# ------------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["Maths"], df["Science"])
plt.title("Maths vs Science")
plt.xlabel("Maths")
plt.ylabel("Science")
plt.grid(True)
plt.show()

# ------------------------------------------
# 6. Histogram
# ------------------------------------------

plt.figure(figsize=(7,5))
plt.hist(df["English"], bins=5)
plt.title("English Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------
# 7. Pie Chart
# ------------------------------------------

plt.figure(figsize=(6,6))
plt.pie(
    df["Maths"],
    labels=df["Student"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Maths Marks Percentage")
plt.show()

# ------------------------------------------
# 8. Box Plot
# ------------------------------------------

plt.figure(figsize=(6,5))
plt.boxplot(df["Science"])
plt.title("Science Marks Box Plot")
plt.ylabel("Marks")
plt.show()

# ------------------------------------------
# 9. Seaborn Line Plot
# ------------------------------------------

sns.lineplot(x="Student", y="English", data=df)

plt.title("English Marks")
plt.show()

# ------------------------------------------
# 10. Seaborn Bar Plot
# ------------------------------------------

sns.barplot(x="Student", y="Maths", data=df)

plt.title("Maths Bar Plot")
plt.show()

# ------------------------------------------
# 11. Seaborn Scatter Plot
# ------------------------------------------

sns.scatterplot(
    x="Maths",
    y="Science",
    data=df
)

plt.title("Maths vs Science")
plt.show()

# ------------------------------------------
# 12. Seaborn Histogram
# ------------------------------------------

sns.histplot(df["Maths"], bins=5)

plt.title("Maths Histogram")
plt.show()

# ------------------------------------------
# 13. Correlation
# ------------------------------------------

print("\nCorrelation Matrix")
print(df[["Maths", "Science", "English"]].corr())

# ------------------------------------------
# 14. Heatmap
# ------------------------------------------

sns.heatmap(
    df[["Maths","Science","English"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# 15. Pair Plot
# ------------------------------------------

sns.pairplot(
    df[["Maths","Science","English"]]
)

plt.show()

# ------------------------------------------
# 16. Average Marks
# ------------------------------------------

df["Average"] = (
    df["Maths"] +
    df["Science"] +
    df["English"]
) / 3

print("\nAverage Marks")
print(df[["Student","Average"]])

# ------------------------------------------
# 17. Highest Average
# ------------------------------------------

highest = df.loc[df["Average"].idxmax()]

print("\nTop Student")
print(highest)

# ------------------------------------------
# END
# ------------------------------------------

print("\nPart 3 Completed Successfully.")
