* Basic Probability Concepts
    * Sample Space and Events 
        * Sample Space: The set of all positive outcome of random experiment
        * Events: A subset of the sample space 
    * Conditional Probability
        * The probability  of an event A occuring, giving that B has occured 
    * Independence 
        * Two events A and B are independent 

* Random Variables
    * What are random variables?
        * Maps outcomesof a random experiment to numerical variables
        * Types: Discrite | Continuous
    * Probability Mass Function(PMF)
        * Probability distribution of a discrete random variable 
    * Pribabiblity Density Function(PDF)
        * Probability distribution of a continuous random variable 

* Expectation, Variance, and Standard deviation
    * Expectation
        * Weighted average of a random  veriable's possible values
    * Variance 
        * Mesuare the spread of a random variable
    * Standard Deviation
        * Square root of variance 

* Common Probability Distribution
    * Gaussian(Normal) Distribution
        * Bell-shaped curve characterized by mean and standard deviation
        * Probability Density Fucntion(PDF)
        * Properties:
            * Symetric about the mean
            * Mean, Median and Mode are the same 
        * Applications in ML are:
            * Common assumptions in many algorithms
            * Used in feature scalling(standarization)
    * Binomial Distribution
        * Models the number of success in (n) independent Bernoulli trials
        * Probability Mass Function(PMF)
        * Properties:
            * Discrete distribution
            * Parameters: (n)-> numbers of trials, (p)-> probability of success
        * Application in ML are:
            * Logistic regression assumes a binomial distribution for binary classification 
    * Poisson Distribution
        * Models the number of events in a fix interval
        * Probability Mass Function(PMF)
        * Properties:
            * Discrete distribution
            * Parameters: (Average rate of occurrence) 
        * Application in ML are:
            * Used in event modeling
    * Uniform Distribution
        * Equal to probability for all outcomes in a range 
        * Probability Density Function(PDF)
        * Properties:
            * Continuous Distribution
            * Parameters: (a)-> lower bound, (b)-> upper bound
        * Application in ML are:
            * Random initialization of weights in neural networks

* Application of Distribution in Machine Learning
    * Gaussian Distribution
        * Used in algorithms like Naive Bayes and Gaussian Mixture Models
        * Assumed in statistical tests
    * Binomial Distribution
        * Foundational for logic regression and other binary classification model 
    * Poisson Distribution
        * Applied in modeling count data
    * Uniform Distribution
        * Commonly used in random sampling and initialization of parameters

* Visualizing Distribution  and Understanding  their properties 
    * Visualization helps understand skewness, kurtosis and outliers
        * Skewness? Measure of Semmetry, Positive skew | Negative skew
        * Kurtosis? Measure of the tailedness of the distribution, High kurtosis | Low kurtosis


* Introduction to Statical Inference
    * What is a Statical inference?
        * Process of making conclusions about population based on sample data
    * Population vs Sample
    * The Goal
        * Estimate population parameters  and assess the reliablity of these estimates 

* Point Estimation and Interval Estimation
    * Point Estimation
        * Single value estimate a population of a parameter
    * Interval Estimation
        * Provides a range of values within which the population parameter is likely to lie
        * Confidence interval (CI)

* Constructing Confidence Intervals
    * For Means
        * When a population standard deviation is unknown 
        * Use the t-distribution for small samples(n < 50)

