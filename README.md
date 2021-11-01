# Offensive Language Detection
Digital bullying is a daily phenomena that each and every one face from time to another.

Proposing a solution that uses machine learning classifiers to detect this offensive language (tweets in our case), and then decide whether it is targeted, and if so, it classifies the target.

## Technologies
- Scikit Learn 0.20
- NLTK 3.4

## Classifiers
- Multinomial Naive Bayes
- Support Vector Machine
- Decision Tree
- Logistic Regression

## How it works
We divide the pre-processing phase into multiple stages, in which we remove stop words, emojis, mentions, urls and all kind of noise, along with a stage of lemmatizing and stemming.

The clean toknized tweets is then sent to TF-IDF vectorizer, that takes care of converting the data into a model of numerical features that are ready to be used for classification.

We apply cross validation on the training vectors with 0.2 splitting factor, while tuning some of the selected parameters to enhance the accuracy.

Finally the best estimator of the selected classifier is used to predict the test labels.

## Results
	Classifing whether the tweet is offensive:
		Training Accuracy : 0.754
		Cross validation Accuracy : 0.76
