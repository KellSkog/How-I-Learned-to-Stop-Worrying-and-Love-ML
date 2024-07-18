# How to Think About Machine Learning Algorithms
## Module 2: Introducing Machine Learning
### Recognizing Machine Learning Applications
### Knowing When to Use Machine Learning
### Understanding the Machine Learning Process
ML Problems generally fall into one of these categories
- Classification<br>
e.g. Sentiment analysis and Spam detection
- Regression<br>
e.g. Sales forecasting, stockmaket value
- Clustering<br>
e.g. Customer segmentation
- Recommendation<br>
e.g. Next movie to watch
Classification and Regression are examples of supervised learning.

Data in the form of unstructured text, images or videos requires a meaningful numeric representation, e.g. 1,2,3,4,5,6,7 to represend days in a week.

Select an algorithm
- Classification
- Naive Bayes
- Support Vector Machine (SVM)

Clustering problems are mainly supported by
- K-Means
- Hierarchicical Clustering

Algorithms can usually be utilized in a Plug-and-Play fashion

Data preparation will consume the lion share of project time.

### Identifying the Type of a Machine Learning Problem
- Classification<br>
We have something that can be classified and categories into which to classify:
    - Is this email SPAM or HAM?
    - Is this tweet positive or negative?
    - Will this trading day close up or down?
- Regression
Compute a continusvalue.
    - What will be the price at any given day?
    - How log time will the commute take?
    - What will the sales be of a broduct a future week?
- Clustering
We have a large number of people that we want to find common denominators of
    - Thiefs
    - Fornicators
    - Murderers
- Recommendation
E.g. what kind of artist will this user like aka collaborative filtering.

![](/HowToThinkAboutMachineLearningAlgorithms/Progress13.png)

## Module 3: Classifying Data into Predefined Categories
### Understanding the Setup of a Classification Problem
- Problem statement
    - A problem instance is a object/entity to be classified.
- Features (Input)
- Lables (Ouput)<br>
The process to follow is:
- Define problem statement
- Represent features in the test and training data numerically
- Train the algorithm to obtain a model
    - Expose the algorithm to correctly labled problem instances.
    - Training data consists of tuples of (Features, Label)
    - Examples of algorithms are
        - Naive Bayes
        - Support Vector Machines
        - Decision Trees
        - K-Nearest Neighbors
        - Random Forests
        - Logistic Regression
- Test the model<br>

We feed the problem instance to a classifier (the model) which assigns a label to the problem instance. (Spam/Cat/Cancer)

### Detecting the Gender of a User
Problem statement
From the first name, classify a user with a label from {Female, Male}.
The name can be given characteristics pertaining to gender (Features):
- Last letter a vowel (1/0)
- Number of characters
- Prefix/suffix attributed to female gender (1/0)
- Prefix/suffix attributed to male gender (1/0)
### Classifying Text on the Basis of Sentiment
- Huge data set =:-o
- Unstructured =:-o
- Freely available :-)<br>
Any big dataset is a learning opportunity - it can be seized via Sentiment Analysis.

e.g. tweets can be grouped on Polarity, as being positive, negative or neutral, so polarity is a possible feature of the dataset.

> Problem statement

Classify a tweet (problem instance) with a label of Positive, Negative or Neutral.

> Features

We need to represent unstructured text with numeric attributes.
Possibilities are
- Term frequence representation<br>
Use a Universe, a list of all possible words e.g. [a, an, aback, abackus, abandon....]
Represent the text of the problem instance with the occurance count e.g.<br>
"abanon an abackus" -> [0, 1, 0, 1, 1, 0....]

### Deciding a Trading Strategy

### Detecting Ads
### Understanding Customer Behavior
![](/HowToThinkAboutMachineLearningAlgorithms/Progress29.png)

## Module 4 Solving Classification Problems
### Using the Naive Bayes Algorithm for Sentiment Analysis
### Understanding When to use Naive Bayes
### Implementing Naive Bayes
### Detecting Ads Using Support Vector Machines
### Implementing Support Vector Machines

## Module 5 Predicting Relationships between Variables with Regression
### Understanding the Regression Setup
### Forecasting Demand
### Predicting Stock Returns
### Detecting Facial Features
### Contrasting Classification and Regression
 
## Module 6 Solving Regression Problems
### Introducing Linear Regression
### Applying Linear Regression to Quant Trading
### Minimizing Error Using Stochastic Gradient Descent
### Finding the Beta for Google
### Implementing Linear Regression in Python
 
## Module 7 Recommending Relevant Products to a User
### Appreciating the Role of Recommendations
### Predicting Ratings Using Collaborative Filtering
### Finding Hidden Factors that Influence Ratings
### Understanding the Alternative Least Squares Algorithm
### Implementing ALS to Find Movie Recommendations

## Module 8 Clustering Large Data Sets into Meaningful Groups
### Understanding the Clustering Setup
### Contrasting Clustering and Classification
### Document Clustering with K-Means
### Implementing K-Means Clustering

## Module 9 Wrapping up and Next Steps
### Surveying Machine Learning Techniques
### Looking Ahead