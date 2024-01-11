# 30000-Spotify-Songs
A model for predicting music genres


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.stats import zscore, ttest_ind, f_oneway
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
df = pd.read_csv('/kaggle/input/30000-spotify-songs/spotify_songs.csv')
df.head()
df.tail()
df.sample()
df.info()
df.dtypes
df.shape
df.columns
df.describe().T
df.isnull().sum()
df.dropna()
df.isnull().sum()
df.isna().any(axis=0)
df['track_name'].fillna('Unknown', inplace=True)
df['track_artist'].fillna('Unknown', inplace=True)
df['track_album_name'].fillna('Unknown', inplace=True)
df.isnull().sum()
df.duplicated()
df[df.duplicated()]
df.hist(figsize=(10,10))
sns.boxplot(x='track_popularity', data=df)
plt.show()
sns.boxplot(x='danceability', data=df)
plt.show()
sns.boxplot(x='energy', data=df)
plt.show()
sns.boxplot(x='key', data=df)
plt.show()
sns.boxplot(x='loudness', data=df)
plt.show()
sns.boxplot(x='mode', data=df)
plt.show()
sns.boxplot(x='speechiness', data=df)
plt.show()
sns.boxplot(x='instrumentalness', data=df)
plt.show()
sns.boxplot(x='liveness', data=df)
plt.show()
sns.boxplot(x='valence', data=df)
plt.show()
sns.boxplot(x='tempo', data=df)
plt.show()
sns.boxplot(x='duration_ms', data=df)
plt.show()
df = df[(df['danceability'] >= 0.28) & (df['danceability'] <= 1)]
sns.boxplot(data=df[['danceability']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['energy'] >= 0.21) & (df['energy'] <=1)]
sns.boxplot(data=df[['energy']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['loudness'] >= -12.5) & (df['loudness'] <=0.3)]
sns.boxplot(data=df[['loudness']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['speechiness'] >=0) & (df['speechiness'] <=0.11)]
sns.boxplot(data=df[['speechiness']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['instrumentalness'] <= 0.000019) & (df['instrumentalness'] >= 0.000001)]
sns.boxplot(data=df[['instrumentalness']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['liveness'] >= 0) & (df['liveness'] <= 0.37)]
sns.boxplot(data=df[['liveness']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['tempo'] >= 53) & (df['tempo'] < 176)]
sns.boxplot(data=df[['tempo']])
plt.title('Boxplot for Trimmed data')
plt.show()
df = df[(df['duration_ms'] >= 120000) & (df['duration_ms'] <= 305000)]
sns.boxplot(data=df[['duration_ms']])
plt.title('Boxplot for Trimmed data')
plt.show()
df.shape
'''scatter_columns = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
sns.pairplot(df[scatter_columns])
plt.show'''
df.playlist_genre.value_counts()
genre_counts = df['playlist_genre'].value_counts()
plt.bar(genre_counts.index, genre_counts.values, color=['#7fcce5', '#ff8c42', '#84a59d', '#cc99c9', '#6e5773', '#d1ae9f'])
plt.xlabel('Music Genre')
plt.ylabel('Number of Songs')
plt.title('Music Genre Distribution')
plt.show()
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=90, colors=['#7fcce5', '#ff8c42', '#84a59d', '#cc99c9', '#6e5773', '#d1ae9f'])
# Adding a title
plt.title('Music Genre Distribution')
# Show the plot
plt.show()
fig,ax=plt.subplots(figsize = (20,5))
plt.bar(df.track_artist.value_counts()[:10].index,df.track_artist.value_counts()[:10].values, color=['#7fcce5', '#ff8c42', '#84a59d', '#cc99c9', '#6e5773', '#d1ae9f'])
plt.xlabel('Artist Name')
plt.ylabel('Number of Listeners')
plt.title('Top 10 Most Listened to Artists on Spotify ')
fig.show()
# top 10 Artists by popularity
df.loc[:,['track_name','track_artist','track_popularity']].sort_values('track_popularity',ascending=False).drop_duplicates()[:10]
numerical_columns = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']

covariance_matrix = df[numerical_columns].cov()

print("Covariance Matrix:")
print(covariance_matrix)
plt.figure(figsize=(20, 15))
sns.set(style="white")  # Set background style
sns.heatmap(covariance_matrix, annot=True, cmap="YlOrBr", fmt=".2f",xticklabels=numerical_columns, yticklabels=numerical_columns)
plt.title("Covariance Matrix Heatmap")
plt.show()
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
categorical_columns = ['track_name', 'track_artist', 'track_popularity','track_album_name','track_album_release_date','playlist_name','playlist_genre','playlist_subgenre']
#comparing track_name and track_artist (change to track_popularity and playlist_genre)
contingency_table = pd.crosstab(df[categorical_columns[0]], df[categorical_columns[1]])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Statistics:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:",expected)
sample_size = 100
df_sample = df.sample(sample_size)
population_mean = df["track_popularity"].mean()
population_std = df["track_popularity"].std()
sample_mean = df_sample["track_popularity"].mean()
alpha = 0.05

# z_test
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
print("Z-Score :", z_score)


z_critical = stats.norm.ppf(1 - alpha)
print("Critical Z-Score :", z_critical)


if z_score > z_critical:
    print("Reject H0")
else:
    print("Fail to Reject H0")
anova_result = f_oneway(df['track_popularity'][df['playlist_genre'] == 'pop'],
                        df['track_popularity'][df['playlist_genre'] == 'latin'],
                        df['track_popularity'][df['playlist_genre'] == 'r&b'],
                        df['track_popularity'][df['playlist_genre'] == 'edm'],
                        df['track_popularity'][df['playlist_genre'] == 'rock'],
                        df['track_popularity'][df['playlist_genre'] == 'rap']
                        )
print("\nANOVA Result")
print("F-statistic:", anova_result.statistic)
print("P-value:", anova_result.pvalue)
# Select features and target variable
features = df[['track_popularity', 'danceability', 'energy', 'key',
                  'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                  'liveness', 'valence', 'tempo', 'duration_ms']]

target = df['playlist_genre']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Applying PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# Applying LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)
# Applying SVD
svd = TruncatedSVD(n_components=2)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
X_train_pca, X_train_lda, X_train_svd
X_train_combined = np.hstack((X_train_pca, X_train_lda, X_train_svd))
X_test_combined = np.hstack((X_test_pca, X_test_lda, X_test_svd))

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
X_train_combined, y_train_encoded = shuffle(X_train_combined, y_train_encoded, random_state=42)
# Plotting PCA
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis')
plt.title('PCA')

# Plotting LDA
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_train_lda[:, 0], y=X_train_lda[:, 1], hue=y_train, palette='viridis')
plt.title('LDA')

# Plotting SVD
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_train_svd[:, 0], y=X_train_svd[:, 1], hue=y_train, palette='viridis')
plt.title('SVD')

plt.tight_layout()
plt.show()
# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_combined, y_train)
y_pred_nb = nb_model.predict(X_test_combined)
pip install pgmpy
'''from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def discretize_and_add_category(df, columns, percentiles, labels):
    for column in columns:
        new_category_column = f'{column}_category'
        df[new_category_column] = pd.cut(df[column], bins=np.percentile(df[column], percentiles), labels=labels)
    return df

# Assuming you have loaded your dataset into df
# Define the variables of interest
audio_features = ['danceability', 'energy', 'tempo']
target_variable = 'playlist_genre'

# Discretize specified columns and add new category columns
percentiles_danceability = [0, 33, 66, 100]
labels_danceability = ['low', 'medium', 'high']
columns_to_discretize = ['danceability', 'energy', 'tempo']

df = discretize_and_add_category(df, columns_to_discretize, percentiles_danceability, labels_danceability)

# Create a Bayesian Network model
bbn_model = BayesianModel([(feature, target_variable) for feature in audio_features] + [(f'{col}_category', col) for col in columns_to_discretize])

# Fit the model using Maximum Likelihood Estimation
data_subset = df[audio_features + [f'{col}_category' for col in columns_to_discretize] + [target_variable]].sample(frac=0.1, random_state=42)
bbn_model.fit(data_subset, estimator=MaximumLikelihoodEstimator)

# Perform inference using Variable Elimination
inference = VariableElimination(bbn_model)

# Example: Query the model to predict 'playlist_genre' based on audio features
query_result = inference.query(variables=[target_variable], evidence={'danceability_category': 'medium', 'energy_category': 'medium', 'tempo_category': 'medium'})
print(query_result)'''
# Decision Tree
dt_model = DecisionTreeClassifier(criterion='entropy',ccp_alpha=0.005,random_state = 42)
dt_model.fit(X_train_combined, y_train)
y_pred_dt = dt_model.predict(X_test_combined)
# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_combined, y_train)
y_pred_lda = lda_model.predict(X_test_combined)
# PCA
pca_model = PCA(n_components=2)
X_train_pca_model = pca_model.fit_transform(X_train_combined)
X_test_pca_model = pca_model.transform(X_test_combined)
# K-NN (using Euclidean distance)
#k-value
knn_model_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model_euclidean.fit(X_train_combined, y_train)
y_pred_knn_euclidean = knn_model_euclidean.predict(X_test_combined)
# K-NN (using Manhattan distance)
knn_model_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_model_manhattan.fit(X_train_combined, y_train)
y_pred_knn_manhattan = knn_model_manhattan.predict(X_test_combined)
# K-NN (using Chebyshev distance)
knn_model_chebyshev = KNeighborsClassifier(n_neighbors=3, metric='chebyshev')
knn_model_chebyshev.fit(X_train_combined, y_train)
y_pred_knn_chebyshev = knn_model_manhattan.predict(X_test_combined)
# Neural Network
model = Sequential()
model.add(Dense(128, input_dim=X_train_combined.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_combined, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)
def evaluate_model_with_overfitting_check(model, X_train, y_train, X_test, y_test):
    # K-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_test, y_test, cv=kfold, scoring='accuracy')
    print(f'K-fold Cross-Validation Accuracy: {cv_results.mean()}')

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{cm}')

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Error rate
    error_rate = 1 - accuracy
    print(f'Error Rate: {error_rate}')

    # Precision
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}')

    # Recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Recall: {recall}')

    # F-measure
    f_measure = f1_score(y_test, y_pred, average='weighted')
    print(f'F-measure: {f_measure}')

    # ROC-AUC (if applicable)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f'ROC-AUC: {roc_auc}')

    # Train set accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    # Determine if the model is overfitting or underfitting
    if train_accuracy > accuracy + 0.1:
        print("Model might be overfitting.")
    elif train_accuracy + 0.1 < accuracy:
        print("Model might be underfitting.")
    else:
        print("Model is likely fitting well.")
def plot_roc(model, X_test, y_test):
    if hasattr(model, 'predict_proba'):
        disp = plot_roc_curve(model, X_test, y_test)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.show()
    else:
        print("ROC curve is not applicable for models that do not have 'predict_proba' method.")

# Example usage:
print("Naive Bayes:")
evaluate_model_with_overfitting_check(nb_model, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(nb_model, X_test_combined, y_test)

print("\nDecision Tree:")
evaluate_model_with_overfitting_check(dt_model, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(dt_model, X_test_combined, y_test)

print("\nLDA:")
evaluate_model_with_overfitting_check(lda_model, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(lda_model, X_test_combined, y_test)

print("\nK-NN (Euclidean):")
evaluate_model_with_overfitting_check(knn_model_euclidean, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(knn_model_euclidean, X_test_combined, y_test)


print("\nK-NN (Manhattan):")
evaluate_model_with_overfitting_check(knn_model_manhattan, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(knn_model_manhattan, X_test_combined, y_test)


print("\nK-NN (Chebyshev):")
evaluate_model_with_overfitting_check(knn_model_chebyshev, X_train_combined, y_train, X_test_combined, y_test)
plot_roc(knn_model_chebyshev, X_test_combined, y_test)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming your dataset is stored in a DataFrame named 'df'
# Drop rows with missing values for simplicity
df_cleaned = df.dropna()

# Encode the target variable 'playlist_genre'
label_encoder = LabelEncoder()
df_cleaned['playlist_genre_encoded'] = label_encoder.fit_transform(df_cleaned['playlist_genre'])

# Select only numeric columns
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
X = df_cleaned[numeric_columns].drop(columns=['playlist_genre_encoded'])  # Exclude the target variable

y = df_cleaned['playlist_genre_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

# Sort features by importance in descending order
sorted_features = feature_importances.sort_values(ascending=False)

# Display the top features
print(sorted_features.head(10))
from sklearn.model_selection import cross_val_score, KFold
#K-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy scores
cv_scores = cross_val_score(dt_model, features, target, cv=kfold, scoring='accuracy')

# Calculate and print the average accuracy
average_accuracy = cv_scores.mean()
print("Average Accuracy:", average_accuracy)
predictions = cross_val_predict(dt_model, features, target, cv=kfold)

class_labels = target.unique()

# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(target, predictions, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
