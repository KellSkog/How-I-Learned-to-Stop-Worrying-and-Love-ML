# How-I-Learned-to-Stop-Worrying-and-Love-ML
My journey from noob to proficient ML coder, picking up a bit of MarkDown on the way :-)<br>
Markdown is like meetings: where minutes are kept, but hours are lost.

# Week 1 Day 1
## Module 2
Lets try: https://www.pluralsight.com/browse/machine-learning

Your Free Trial begins today and will expire in 10 days (up to 200 minutes). You can convert your free trial to a paid subscription in your account settings at any time before your trial runs out.
You will not be charged until your free trial ends. Your free trial will renew as the Individual Premium (Monthly) plan for 45 US$✝ on Thursday, July 25, 2024.

The initial assessment seems about right

![](/Week%201/Day%201/Screenshot%202024-07-15%20090644.png)

### Types of algorithms
1. Supervised Learning
    - Model created from historical data and outcome  
    Predicts future value or category from historic outcome.
        1. Regression
            - Can result in any value e.g. the amount of water to supply to plants.
        1. Classification
            - Can result in a limited set of outputs (Cat, Dog, COBOL coder)
1. Unsupervised learning - Not part of this course!
    - Model created from historical data without outcome  
    Organize data based on characteristics in input data.
        1. Clustering
            - Spots similarities
        1. Association
            - Finds association between variables
        1. Reinforsement
1. Reinforcement learning aka goal oriented - Advanced topic outside this course.  
Adapts to rewards received based on actions changing the state of the environment.
    - Creates agents
    The agents act on the environment and recieves rawards based on outcome  
    and learns to act according to best reward.

### Learning modes
1. Batch
1. Online

### ML Pipeline
1. Problem definition
1. Data sourcing, Preparation & segregation
1. Model training & evaluation
1. Model deployment & monitoring

### Forest Fire Forecast
forestfires.csv  
Problem definition: predict area burnt by supervised regression learning.  
Require background in linear algebra, statistics and probability.

Google [Colaboratory](https://colab.google/)
> Colab is a hosted Jupyter Notebook service that requires no setup to use and provides free access to computing resources, including GPUs and TPUs. Colab is especially well suited to machine learning, data science, and education.

First notebook created: My First ML solution.ipynb

![](/Week%201/Day%201/Progress17.png)

## Module 3
As data sourcing is where a majority of project time is invested normally,
here we focus on simple CSV to focul on actual ML.

### SciPy package
> What is SciPy?

SciPy is a set of open source (BSD licensed) scientific and numerical tools for Python. It currently supports special functions, integration, ordinary differential equation (ODE) solvers, gradient optimization, parallel programming tools, an expression-to-C++ compiler for fast execution, and others. A good rule of thumb is that if it's covered in a general textbook on numerical computing (for example, the well-known Numerical Recipes series), it's probably implemented in SciPy.

Copilot describes SciPy's relation with Numpy, Matplotlib and Pandas thus:

> **SciPy** is a powerful **Python library** that builds upon other libraries to provide a comprehensive suite of scientific and technical tools. Here's how it relates to the ones you mentioned:
>
> 1. **NumPy**: **SciPy** heavily relies on **NumPy**. **NumPy** provides fundamental array operations, linear algebra, and numerical routines. **SciPy** extends these capabilities by adding specialized functions for optimization, integration, interpolation, and more.
> 2. **Matplotlib**: While **SciPy** itself doesn't directly include **Matplotlib**, they are often used together. **Matplotlib** is a plotting library for creating visualizations, and it integrates seamlessly with **SciPy**. You can use **SciPy** to compute data, and then visualize it using **Matplotlib**.
> 3. **Pandas**: **Pandas** is a separate library for data manipulation and analysis. It's not part of **SciPy**, but it complements it well. **SciPy** focuses on scientific and mathematical functions, while **Pandas** provides powerful data structures (like DataFrames) and tools for data cleaning, exploration, and transformation.
>
> In summary, **SciPy** doesn't directly contain **NumPy**, **Matplotlib**, or **Pandas**, but it works closely with them to enhance Python's capabilities for scientific computing and data analysis!

A bit of maintenance:

> python -m pip install --upgrade pip  
> python -m pip install scipy  
> pip install matplotlib==3.9.0  
> pip install pandas
> python -m pip install seaborn

tryNump.py demos numpy, matplotlib and pandas loading forestfire.csv

![](/Week%201/Day%201/Progress30.png)

## Module 4
> In this module, we will continue with more detailed data preparation.
We will do data analysis to understand the datas shape, its relationships, its mathematical characteristics and visualize it. This will help us to make a more informed decision around the usage of the right machine learning algorithms, among other benefits. We need to analyze our data as a part of the preparation process for the subsequent steps in the machine learning pipeline. This data analysis is also called data science.
### Revisiting ML Pipeline
> data preparation.
### Introducing Data Analysis
> Data analysis is the process of inspecting, cleansing, transforming, and modeling data with the goal of discovering useful information and informing conclusion and supporting decision‑making
Wikipedia.
- Identify dataset distribution.
- Choosing right ML algorithm.
- Extracting the right features.
- Different models needs to be trained at the same time and select the best performing.
- Presentation of result.

> Exploratory data analysis
- Numerical summaries e.g. mean or average
- Graphical summaries e.g. histogram

### Univariant Numerical Analysis
This section will introduce the numerical summaries and will be more or less a refresher to the basic statistics.<br>
Univariant measures refer to measures that rely only on a single variable,

### Bivariant Numerical Analysis
- Correlation, measure of extent of linear relativity<br>
[Correlation Coefficients: Positive, Negative, and Zero](https://www.investopedia.com/ask/answers/032515/what-does-it-mean-if-correlation-coefficient-positive-negative-or-zero.asp)
$$
Cor(x,y) = \frac{\Sigma(\bar{x}-u_x)(\bar{y}-u_y)}{\sqrt{\Sigma(\bar{x}-u_x)^2(\bar{y}-u_y)^2}}
$$<br>
Looked so good in the VS Code<br>
![](/Week%201/Day2/CorrelationCoeff.png)

Beware of correlation fallacy, correlation does not imply causation!

[Pandas Options and settings](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)<br>
[pandas.DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)

### Demo: Descriptive Stats - Part One

### Demo: Descriptive Stats - Part Two
Here I run into problems with
> print(f"Pearson\n{dataFrame.corr(method='pearson')}")<br>
> ValueError: could not convert string to float: 'mar'

While the course video shows there is no problem, and susequently fixes the problem...???<br>
Wow that was a bit of a detour<br>
Deprecation had to be handled "pd.set_option('future.no_silent_downcasting', True)"<br>
.replace did not work as expected, it did not return the dataFrame object as expected with 'inplace=False'

The replacements done here are not a good solution, there are better methods that should be used, but are left out of this course!!

### Data Visualization
Graphical summaries
Distributions
- Normal, Gaussian, bell shaped
- Skewed, the peak is shifted to one side.
- Exponential

Helps detect impossible values, identify the data shape and errors in the data.<br>

Box and Whisker plot:<br>
![](/Week%201/Day2/BoxAndWiskerPlot.png)

IQR (Inter Quartal Range) the interval between the upper and lower quartile.
Upper extreme = Upper quartile + 1.5IQR<br>
Lower extreme = lower quartile - 1.5IQR

### Demo: Data Visualization - Part One
Oh wow, more friction:
> FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
>
>For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.

With this warning here are the histograms of the forestfire data:
![](/Week%201/Day2/Histograms.png)

![](/Week%201/Day2/pdf.png)

![](/Week%201/Day2/box.png)

### Demo: Data Visualization - Part Two
![](/Week%201/Day2/scatter.png)

Alas seaborn library "Successfully installed seaborn-0.13.2"<br>
Failed to produce a plot :-c
Ah, the video was missing the all essential plt.show() :-D<br>
![](/Week%201/Day2/Heatmap.png)

![](/Week%201/Day2/Progress50.png)

## Making Your Data Ready for the ML Model

### Revisiting ML Pipeline
### Data Scaling: The problem
### Data Scaling: The Solution
### The need for data segregation
### Train Test Split
### KFlod Cross Validation
### Welcoming scikit-learn
### Demo: Data Segregation Techniques
