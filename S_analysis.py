import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from imblearn.combine import SMOTETomek
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, \
    precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import re, warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Loading the dataset and checking if file exists
file_path = 'Apple_iPhone_review.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
df = pd.read_excel(file_path)

# Shuffle dataset
df = df.sample(frac=1)

# Data cleaning and Preprocessing *****************************

# Customize stopwords
stop_words = stopwords.words('english')
new_stopwords = {"phone", "iphone", "mobile", "flipkart", "would", "shall", "could", "might"}
stop_words = set(stop_words).union(new_stopwords)
stop_words.discard("not")


# Data cleaning functions

# Removing special characters
def remove_special_characters(content):
    return re.sub(r'[^A-Za-z0-9\s]', '', content)


# Removing URLs
def remove_url(content):
    return re.sub(r'http\S+', '', content)


# Removing stopwords from the text
def remove_stopwords(content):
    clean_data = []
    for word in content.split():
        word_lower = word.strip().lower()
        if word_lower not in stop_words and word_lower.isalpha():
            clean_data.append(word_lower)
    return " ".join(clean_data)


# Expansion of English contractions
def contraction_expansion(content):
    contractions = {
        "won't": "will not", "can't": "can not", "don't": "do not",
        "shouldn't": "should not", "wouldn't": "would not",
        "needn't": "need not", "haven't": "have not", "hasn't": "has not",
        "weren't": "were not", "mightn't": "might not", "didn't": "did not",
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        content = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, content)
    return content


# Data Preprocessing
def data_cleaning(content):
    if not isinstance(content, str):
        content = str(content) if not pd.isna(content) else ""
    content = contraction_expansion(content)  # Expand contractions
    content = remove_special_characters(content)  # Remove special characters
    content = remove_url(content)  # Remove URLs
    content = remove_stopwords(content)  # Remove stopwords
    return content


# Handle missing values before cleaning
df['Review'] = df['Review'].fillna("")  # Replace NaN with an empty string

# Data cleaning ************************************
df['Review'] = df['Review'].apply(data_cleaning)

print(df['Review'].head(5))

# Checking for missing values
print(df.isna().sum())

# Statistical Analysis ************************************
print(df['Rating'].describe())
print(df['Review'].describe())

print("Unique reviews:{}".format(df['Review'].nunique()))
print("Unique Phone names:{}".format(df['Phone'].nunique()))
print("No. of ratings:{}".format(df['Rating'].count()))

# Mapping the rating to binary label
df['Label'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
data = df[['Review', 'Label']]
print(data['Label'].value_counts())

# Creating a new column 'length' that will contain the length of the string in 'verified_reviews' column
df['Length'] = df['Review'].apply(len)

# Distinct values of 'rating' and its count
print(f"Rating value count: \n{df['Rating'].value_counts()}")

# print(df.columns.tolist())

# Exploratory Data Analysis ************************************

# Bar plot to visualize the total counts of each rating
df['Rating'].value_counts().plot.bar(color='crimson')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

# Finding the percentage distribution of each rating
print(f"Rating value count - percentage distribution: \n{round(df['Rating'].value_counts() / df.shape[0] * 100, 2)}")

# Pie chart of rating percentage
fig = plt.figure(figsize=(7, 7))

colors = ('violet', 'pink', 'turquoise')

wp = {'linewidth': 1, "edgecolor": 'yellow'}

tags = df['Rating'].value_counts() / df.shape[0]

explode = [0.1] * len(tags)

tags.plot(
    kind='pie',
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops=wp,
    explode=explode,
    label='Percentage wise distribution of rating'
)

plt.ylabel("")
plt.title("Rating Percentage Distribution")
plt.show()

# Distinct values of 'feedback' and its count
print(f"Label value count: \n{df['Label'].value_counts()}")

# Bar graph to visualize the total counts of each feedback
df['Label'].value_counts().plot.bar(color='teal')
plt.title('Label distribution count')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Finding the percentage distribution of each label
print(f"Label value count - percentage distribution: \n{round(df['Label'].value_counts() / df.shape[0] * 100, 2)}")

fig = plt.figure(figsize=(7, 7))

colors = ('lavender', 'purple')

wp = {'linewidth': 1, "edgecolor": 'yellow'}

tags = df['Label'].value_counts() / df.shape[0]

explode = [0.1] * len(tags)

tags.plot(kind='pie', autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops=wp, explode=explode,
          label='Percentage wise distrubution of the labels')

plt.ylabel("")
plt.title("Label Percentage Distribution")
plt.show()

# Label = 1
print(df[df['Label'] == 1]['Rating'].value_counts())

# Label = 0
print(df[df['Label'] == 0]['Rating'].value_counts())

# Types of 'phone' and its count
print(f"Phone type value count: \n{df['Phone'].value_counts()}")

# Bar graph to visualize the total counts of each phone type
df['Phone'].value_counts().plot.barh(color='orange')
plt.title('Phone type distribution count')
plt.xlabel('Count')
plt.ylabel('Phone')
plt.show()

# Finding the percentage distribution of each phone type
print(f"Phone type value count - percentage distribution: \n{round(df['Phone'].value_counts() / df.shape[0] * 100, 2)}")
print(df.groupby('Phone')['Rating'].mean())

# Analyze the above ratings
df.groupby('Phone')['Rating'].mean().sort_values().plot.barh(color='brown', figsize=(11, 6))
plt.title("Mean rating according to variation")
plt.xlabel('Mean rating')
plt.ylabel('Phone')
plt.show()

# Analyzing 'Reviews' column

print(df['Length'].describe())

# Length analysis for full dataset
sns.histplot(df['Length'], color='blue').set(title='Distribution of length of review ')
plt.show()

# Length analysis when label is 0 (negative)
sns.histplot(df[df['Label'] == 0]['Length'], color='crimson').set(title='Distribution of length of review if label = 0')
plt.show()

# Length analysis when feedback is 1 (positive)
sns.histplot(df[df['Label'] == 1]['Length'], color='green').set(title='Distribution of length of review if label = 1')
plt.show()

# Lengthwise mean rating
df.groupby('Length')['Rating'].mean().plot.hist(color='cyan', figsize=(7, 6), bins=20)
plt.title(" Review length wise mean ratings")
plt.xlabel('Ratings')
plt.ylabel('Length')
plt.show()

# CountVectorizer
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(df.Review)

# Combine all reviews
reviews = " ".join([review for review in df['Review']])

# Initialize Wordcloud Object for all
wc = WordCloud(width=1500, height=800,
                          background_color='black',
                          stopwords=stop_words,
                          min_font_size=15).generate(reviews)

# Generate and plot wordcloud
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()

# Initialize Wordcloud Object for each label
sentences = df['Review']
pos = ' '.join(map(str, sentences[df['Rating'] >= 4]))
neg = ' '.join(map(str, sentences[df['Rating'] < 4]))

# Wordcloud of important words from positive reviews
pos_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                          stopwords=stop_words,
                          min_font_size=15).generate(pos)

plt.figure(figsize=(10, 10))
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.title('Positive Reviews WordCloud')
plt.axis('off')
plt.show()

# Wordcloud of important words from negative reviews
neg_wordcloud = WordCloud(width=1500, height=800,
                          background_color='black',
                          stopwords=stop_words,
                          min_font_size=15).generate(neg)

plt.figure(figsize=(10, 10))
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.title('Negative Reviews WordCloud')
plt.axis('off')
plt.show()

# Preprocessing and modelling ************************************

# Building corpus
corpus = []
stemmer = PorterStemmer()
for i in range(0, df.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', df.iloc[i]['Review'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)

# Using Count Vectorizer to create bag of words
cv = CountVectorizer(max_features = 2500)

# Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = df['Label'].values

# Saving the Count Vectorizer
os.makedirs('Models', exist_ok=True)
with open('Models/countVectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)


# Checking the shape of X and y
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Splitting data into train and test set with 30% data with testing.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 15
                                                    )
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")
print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")


# We'll scale X_train and X_test so that all values are between 0 and 1.
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Apply SMOTE-Tomek to handle class imbalance
smote_tomek = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_scl, y_train)

# Verify the new class distribution
print("Class distribution after SMOTE-Tomek:", pd.Series(y_train_bal).value_counts())

# Saving the scaler model
with open('Models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Random Forest_________________________________

# Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier(class_weight="balanced")
model_rf.fit(X_train_bal, y_train_bal)

# Threshold tuning for Random Forest
y_preds_rf = model_rf.predict(X_test_scl)
y_probs_rf = model_rf.predict_proba(X_test_scl)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_rf)
optimal_threshold = thresholds[np.argmax(recalls - precisions)]
y_preds_rf_adjusted = (y_probs_rf >= optimal_threshold).astype(int)


# Accuracy of the model on training and testing data
print("Random Forest accuracies - ")
print("Training Accuracy :", model_rf.score(X_train_bal, y_train_bal))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))
print("Classification Report:\n", classification_report(y_test, y_preds_rf_adjusted, zero_division=0))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs_rf))


# Confusion Matrix
# cm = confusion_matrix(y_test, y_preds_rf_adjusted)
# cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
# cm_display.plot()
# plt.show()


# K fold cross-validation ____________________________________
accuracies = cross_val_score(estimator = model_rf, X = X_train_bal, y = y_train_bal, cv = 10)
print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())

# Applying grid search to get the optimal parameters on random forest
params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}

cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_bal, y_train_bal.ravel())

# Getting the best parameters from the grid search
print("Best Parameter Combination : {}".format(grid_search.best_params_))
print("K fold cross-validation accuracies - ")
print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
print("Accuracy score for test set :", accuracy_score(y_test, y_preds_rf))

# XG Boost_________________________________

model_xgb = XGBClassifier(scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])))
model_xgb.fit(X_train_bal, y_train_bal)

# Threshold tuning for XGBoost
y_preds_xgb = model_rf.predict(X_test_scl)
y_probs_xgb = model_rf.predict_proba(X_test_scl)[:, 1]
precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_test, y_probs_xgb)
optimal_threshold_xgb = thresholds_xgb[np.argmax(recalls_xgb - precisions_xgb)]
y_preds_xgb_adjusted = (y_probs_xgb >= optimal_threshold_xgb).astype(int)

# Accuracy of the model on training and testing data
print("XG Boost accuracies - ")
print("Training Accuracy :", model_xgb.score(X_train_bal, y_train_bal))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))
print("Classification Report:\n", classification_report(y_test, y_preds_xgb_adjusted, zero_division=0))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs_xgb))

# Confusion Matrix
# cm = confusion_matrix(y_test, y_preds_xgb_adjusted)
# print(cm)
# cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
# cm_display.plot()
# plt.show()

# Saving the XGBoost classifier
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))

# Decision Tree Classifier_________________________________
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_bal, y_train_bal)

# Threshold tuning for Decision Tree
y_probs_dt = model_dt.predict_proba(X_test_scl)[:, 1]
precisions_dt, recalls_dt, thresholds_dt = precision_recall_curve(y_test, y_probs_dt)
optimal_threshold_dt = thresholds_dt[np.argmax(recalls_dt - precisions_dt)]
y_preds_dt_adjusted = (y_probs_dt >= optimal_threshold_dt).astype(int)

# Accuracy of the model on training and testing data
print("Decision Tree Classifier accuracies - ")
print("Training Accuracy :", model_dt.score(X_train_bal, y_train_bal))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))
print("Classification Report:\n", classification_report(y_test, y_preds_dt_adjusted, zero_division=0))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs_dt))

# Confusion Matrix
# cm = confusion_matrix(y_test, y_preds_dt_adjusted)
# print(cm)
# cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
# cm_display.plot()
# plt.show()















