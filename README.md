# How-I-Learned-to-Stop-Worrying-and-Love-ML
My journey from noob to proficient ML coder, picking up a bit of MarkDown on the way :-)

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
Predict area burnt by supervised regression learning.  
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

tryNump.py demos numpy, matplotlib and pandas loading forestfire.csv

![](/Week%201/Day%201/Progress30.png)

##Module 4