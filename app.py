import nltk
import numpy as np
import pandas as pd
from textblob import Word
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from flask import Flask, render_template

app = Flask(__name__)

test = pd.read_csv('jobs-title-and-description.csv')
test = test.dropna()
print("\n ** raw data **\n")
print(test.head())
print("\n ** data shape **\n")
print(test.shape)

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

# Lower case
test['description'] = test['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# remove tabulation and punctuation
test['description'] = test['description'].str.replace('[^\w\s]', ' ')
# digits
test['description'] = test['description'].str.replace('\d+', '')

# remove stop words
stop = stopwords.words('english')
test['description'] = test['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# lemmatization
test['description'] = test['description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Converting text to features
vectorizer = TfidfVectorizer()
# Tokenize and build vocabulary
X = vectorizer.fit_transform(test.description)
y = test.job_title

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)

# Fit model
clf = MultinomialNB()
clf.fit(X_train, y_train)
# Predict
y_predicted = clf.predict(X_test)

technical_skills = ['python', 'c', 'r', 'c++', 'java', 'hadoop', 'scala', 'flask', 'pandas', 'spark',
                    'scikit-learn',
                    'numpy', 'php', 'sql', 'mysql', 'css', 'mongdb', 'nltk', 'fastai', 'keras', 'pytorch',
                    'tensorflow',
                    'linux', 'Ruby', 'JavaScript', 'django', 'react', 'reactjs', 'ai', 'ui', 'tableau']

feature_array = vectorizer.get_feature_names()
# number of overall model features
features_numbers = len(feature_array)
# max sorted features number
n_max = int(features_numbers * 0.1)


@app.route('/')
def index():
    output = pd.DataFrame()
    for i in range(0, len(clf.classes_)):
        print("\n****", clf.classes_[i], "****\n")
        class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
        raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
        print("list of unprocessed skills :")
        print(raw_skills)

        # Extract technical skills
        top_technical_skills = list(set(technical_skills).intersection(raw_skills))[:6]

        # transform list to string
        txt = " ".join(raw_skills)
        top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:6]

        output = output.append({'job_title': clf.classes_[i],
                                'skills': top_adjectives},
                               ignore_index=True)


        # return output.T

    out = output.to_json(orient='records')[1:-1].replace('},{', '} {')
    print(out)

    return render_template('index.html', column_names=output.columns.values, row_data=list(output.values.tolist()),
                               zip=zip)


if __name__ == "__main__":
    app.run(debug=True)
