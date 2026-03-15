* Introduction to Hypothesis Testing 
    * What is Hypothesis Testing?
        * Statistical method to determine is there is enough evidence in a sample to infer a conclusion about the population 
    * Key Components
        * Null Hypothesis: Assumes no effect or no difference
        * Althernative Hypothesis: indicate a effect or difference
    * Steps in Hypothesis Testing
        * Formuate Null Hypothesis and Alternative Hypothesis
        * Choose a significate level(a) - common value are 0.05, or 0.01
        * Calculate the test statistic
        * Determine the p-value
        * Compare the p-value to (a):
            * if p < A -> reject null hypothesis or if p > A -> fail to reject null hypothesis

* Understande P-values and Significance Levels
    * P-value
        * The probability of observing results as extreme as the test statistic under null hypothesis
        * Smaller p-values indicate stronger evidence against null hyphothesis 
    * Significance level(a)
        * Threshold for deciding wheather to reject
        * Example: a=0.05 means a 5% risk of rejecting null hypothesis when is true
    * Decision Rules
        * Rejection Null Hypothesis
        * Fail to Reject Null Hypothesis

    * Types of Errors
        * Type | Error(a)
            * Incorrectly rejecting null hypothesis when its true
            * Example: Concluding a drug is effective when it is not 
        * Type | Error(b)
            * Failing to reject null hypothesis when it is false
            * Example: Concluding a drug is not effective when it is  


* T-Tests
    * Purpose: Test weather the means of one or more groups differ significantly
    * Types
        * One-Sample T-Test: if the mean of sample differs from a known value or population mean
        * Two-Sample T-Test(Independent T-Test): Compare the means of two independent groups
        * Paired Sample T-Test: Compares the means of two related groups(pre-test vs post-test)  
    * Example of usage:
        * One Sample: Testing is the average test score of a class differs  from the national average
        * Two Sample: Comparing test scores between two classes

* Chi-Square Test
    * Purpose: Test for independence or goodness of-fit in categorical data
    * Chi-Square Test for independence: Tests if two categorical variable are independent
    * Example of usage:
        * Testing if gender is independent of preference for a product
            * Steps 
                * Create a contengency table 
                * Calculate expected frequencies
                * Compute X**2 statistic and p-value 

* Anova(Analysis of Variance)
    * Purpose: Compare the means of three or more groups 
    * Hypothesis
        * Null: all groups means are equal
        Alternative: At least one group mean is different
    * Example Use Case: Testing if the mean  score of students from three different schools differs

* Understanding Correlation
    * What is Correlation?
        * Measure  the strength and direction of the  relationship between two variables
        * Values range from 1 to 1, with 0 indicating no correlation 
    * Types of Correlation
        * Pearson Correlation Coefficient(r)
        * Spearman Correlation Coefficient(p)  

* Linear regression Basics
    * What is Linear Regression?
        * A Method to model the relationship between a dependent variable (y), and one or more independent vairable (x)

* Interpreting Regression Results
    * Slope:
        * indicate the magnitude and direction of the relationship
    * Intercept:
        * Starting point od the regression line 
    * R-Squared:
        * Closer to 1 indicates better fit

                






