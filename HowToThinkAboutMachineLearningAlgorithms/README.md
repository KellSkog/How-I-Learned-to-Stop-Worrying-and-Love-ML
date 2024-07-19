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

e.g. tweets can be grouped on Polarity, as being positive, negative, so polarity is a possible feature of the dataset.

> Problem statement

Classify a tweet (problem instance) with a label of Positive, Negative or.

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
The training data consists of labeled Term Frequncy Representation of comments.
A ranomly picked comment has the probability of being positive = P0 (55%) & negative = 1 - P0 (45%), i.e. in this example 55% ov the comments are positive.<br>
A positivity score is computed for every word i.e.<br>
Pos<sub>Happy</sub> = Count of positive "Happy" / Count of all "Happy"<br>
Neg<sub>Happy</sub> = 1 - Pos<sub>Happy</sub><br>
P0 = count of positive words / count of words<br>

The score of a comment (problem instance) is given by:<br>
Pos<sub>comment</sub> = P0 * Pos<sub>word1</sub> * Pos<sub>word2</sub> *...<br>
Neg<sub>comment</sub> = (1 - P0) * (1 - Pos<sub>word1</sub>) * (1 - Pos<sub>word2</sub>) *...<br>
Label the comment as positive if Pos<sub>comment</sub> is greater than Neg<sub>comment</sub> <br>

### Understanding When to use Naive Bayes
The Naive in Naive Bayes signifies that no context is considered, there is an assumption of independence. "Not bad" does not imply good.<br>
However it still performs well, particularly with limited availability of training data.
### Implementing Naive Bayes
[Download](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
Ok the age of this course is beginning to show its age<br>
expected:<br>
![](/HowToThinkAboutMachineLearningAlgorithms/train_documents.png)<br>
But it is:<br>
> <Compressed Sparse Row sparse matrix of dtype 'int64'
        with 31580 stored elements and shape (3000, 5159)><br>


### Detecting Ads Using Support Vector Machines
A Support Vector Machine is used to find the bundary plane between sets of points. The points represent objects in an N-dimensional hypercube. It can only distiguish between two lables: either above or below the boudary.
### Implementing Support Vector Machines
[Download](https://archive.ics.uci.edu/dataset/51/internet+advertisements)<br>
ads.data is 10 MB and excluded from repository.
In ads.data images are represented by over 1500 features and labeled Ad/Non-Ad.<br>
[Exercises in solutions.py](/HowToThinkAboutMachineLearningAlgorithms/src/solutions.py)<br>
![](/HowToThinkAboutMachineLearningAlgorithms/Progress43.png)

## Module 5 Predicting Relationships between Variables with Regression
### Understanding the Regression Setup
Input is called 'Independent variables' and output 'Dependent variables'.<br>
Types of regression
- Linear
- Polynomial
- Non-linear


### Forecasting Demand

### Predicting Stock Returns
Using the Capital Asset Pricing Model, CAPM - used for pricing risky securities<br>
![](/HowToThinkAboutMachineLearningAlgorithms/CAPM.png)

### Detecting Facial Features

### Contrasting Classification and Regression
In classification the training data is used to build a classifier<br>
In regression the training data establish a relation between variables<br>
![](/HowToThinkAboutMachineLearningAlgorithms/Progress56.png)
## Module 6 Solving Regression Problems
- Applying Linear reression to Beta of a stock.
- Stochastic Gradient Method for Linear Regression.
- Tweek SGD parameters for better performance.
- Implement Linear Regression in Python<br>

### Introducing Linear Regression
In case of forecasting Sales an approach could be:<br>
Sales = 2 * Marketing Spend + 0.5 * Last weeks sale<br>
### Applying Linear Regression to Quant Trading
Manipulating CAPM to get profit of a security: R<sub>i</sub> - R<sub>f</sub><br>
We can plot this gain (beta) against some imaginative market gain
as a simple 1-variable linear regression:<br>
![](/HowToThinkAboutMachineLearningAlgorithms/CAPM%20beta.png)<br>

The distance between an individual point and the regression line is the Error, aka Residuals.

### Minimizing Error Using Stochastic Gradient Descent
![](/HowToThinkAboutMachineLearningAlgorithms/SGD%20error.png)<br>

### Finding the Beta for Google
### Implementing Linear Regression in Python
mmmm...
>ValueError: Input X contains NaN.<br>
There were quite some mangling needed for this course written in Python 2, but I got it to work :-D<br>

beta [0.26152312]<br>
> $\>python .\HowToThinkAboutMachineLearningAlgorithms\src\solutions.py google<br>
![](/HowToThinkAboutMachineLearningAlgorithms/Progress70.png)
 
## Module 7 Recommending Relevant Products to a User
### Appreciating the Role of Recommendations
27 Sep 2016

### Predicting Ratings Using Collaborative Filtering
Predict user ratings for products based on the users past behavior<br>
A general term for any algorithm that relies solely on past behavior to produce recommendations.<br>
The basic premise is that other users have the same opinion about products<br>
they likely share opinion of other products too!<br>
### Finding Hidden Factors that Influence Ratings
![](/HowToThinkAboutMachineLearningAlgorithms/MovieRecommendation.png)<br>
![](/HowToThinkAboutMachineLearningAlgorithms/LatentFactorAnalysis.png)<br>
![](/HowToThinkAboutMachineLearningAlgorithms/MinimizeError.png)<br>
### Understanding the Alternative Least Squares Algorithm
ALS is a technique to minimize the error.<br>
It searches for a solution to (r<sub>ui</sub> - q<sub>i</sub><sup>T</sup>p<sub>u</sub>)<sup>2</sup> = 0<br>
by alternatingly fixing q<sub>i</sub> and solving for </sup>p<sub>u</sub>,<br>
with fixing </sup>p<sub>u</sub> and solving for q<sub>i</sub>.
### Implementing ALS to Find Movie Recommendations
Wow this was quite a bit of work compesating for API change of "implicit"<br>
Hot movies [17, 89, 534]<br>
![](/HowToThinkAboutMachineLearningAlgorithms/Progress83.png)

## Module 8 Clustering Large Data Sets into Meaningful Groups
### Understanding the Clustering Setup
### Contrasting Clustering and Classification
### Document Clustering with K-Means
### Implementing K-Means Clustering

## Module 9 Wrapping up and Next Steps
### Surveying Machine Learning Techniques
### Looking Ahead