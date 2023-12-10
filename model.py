import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# load the cleaned dataset
df = pd.read_csv('C:\\Users\\Nghia\\OneDrive\\Documents\\School\\Fall2023\\CDS 303\\Assignments\\final\\cleaned_covid_data.csv')

# select the features and target variables
X = df.drop('DEATH', axis=1)
y = df['DEATH']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create a logistic regression model
LR_model = LogisticRegression()

# fit the model to the training data
LR_model.fit(X_train_scaled, y_train)

# save the model
pickle.dump(LR_model, open('C:\\Users\\Nghia\\OneDrive\\Documents\\School\\Fall2023\\CDS 303\\Assignments\\final\\model.pkl', 'wb'))

# save the scaler
pickle.dump(scaler, open('C:\\Users\\Nghia\\OneDrive\\Documents\\School\\Fall2023\\CDS 303\\Assignments\\final\\scaler.pkl', 'wb'))
