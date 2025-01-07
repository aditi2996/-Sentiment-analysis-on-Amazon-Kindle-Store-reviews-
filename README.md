# Sentiment-analysis-on-Amazon-Kindle-Store-reviews
# Summary
Led sentiment analysis on 982,619 Amazon Kindle Store reviews by deploying advanced NLP techniques. Applied lemmatization and split the dataset into training and testing sets, engineering robust features using Bag of Words (BOW), TF-IDF, and Word2Vec embeddings. Optimized model performance by leveraging TF-IDF and BOW with a range of classifiers, including GaussianNB (Gaussian Naïve Bayes),BalancedRandomForestClassifier, Logistic Regression, and SGD Classifier, achieving a test log loss of 0.27. Delivered a maximum accuracy of 85% with BalancedRandomForestClassifier, coupled with exceptional precision (0.91) and recall (0.88), ensuring highly effective sentiment classification.

# Dataset: 
Processed a dataset of 982,619 entries of Amazon Kindle Store product reviews collected from May 1996 to July 2014. Ensured quality by including only products and reviewers with at least 5 reviews each.
Key Steps and Techniques:
1.	Preprocessing & Cleaning:
o	Applied Lemmatization to normalize text.
o	Split data into training and testing sets.

2.	Feature Engineering:
o	Implemented Bag of Words (BOW) and TF-IDF Vectorizer to represent text.
o	Used Word2Vec embeddings for semantic understanding.

3.	Model Training & Evaluation:
o	BOW: Achieved 62.7% accuracy using GaussianNB.
o	TF-IDF: Achieved 62.35% accuracy with GaussianNB.
o	BalancedRandomForestClassifier: Achieved 85% accuracy on TF-IDF, with precision, recall, and F1 scores as follows:
	Class 0: Precision: 0.77, Recall: 0.88, F1: 0.82 (781 reviews).
	Class 1: Precision: 0.91, Recall: 0.83, F1: 0.87 (1,219 reviews).

4.	CalibratedClassifierCV & Logistic Regression:
o	Logistic Regression:
	Train Accuracy: 84.0%
	Test Accuracy: 82.25%
	Cross-Validation Accuracy: 84.125%
o	CalibratedClassifierCV: Overall accuracy of 82%, with:
	Precision: 0.85 (False), 0.78 (True)
	Recall: 0.86 (False), 0.76 (True)

5.	SGDClassifier with Log Loss:
o	TF-IDF Best Alpha = 0.0001:
	Train Log Loss: 0.1900
	Test Log Loss: 0.2723
	Cross-Validation Log Loss: 0.2878
o	Word2Vec Best Alpha = 1e-05:
	Train Log Loss: 0.3532
	Test Log Loss: 0.3783
	Cross-Validation Log Loss: 0.3711
# Impact:
Demonstrated expertise in text preprocessing, feature extraction, and advanced classification techniques. Achieved a maximum accuracy of 85% using BalancedRandomForestClassifier and high precision and recall scores, showcasing the ability to build robust models for large-scale sentiment analysis.

