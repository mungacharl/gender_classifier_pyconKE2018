from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Training Data [H, W, S]
X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
              [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])

Y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
              'female', 'male', 'male'])

# Create classifiers
decision_tree_classifier = tree.DecisionTreeClassifier()

# Train our models
decision_tree_classifier.fit(X, Y)

# perform prediction
decision_tree_classifier_prediction = decision_tree_classifier.predict([[130, 62, 40]])


# print our prediction
print(decision_tree_classifier_prediction)
