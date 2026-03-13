"""
* Understanding Integrals and their applications in ML 
    * what are integrals ?
        * Compute the area under the curve, representing accumutation
        * Defined as integral of f(x) from A to B
    * Applications on ML
        * Probability distribution
        * Cost Functions     
    
"""
import sympy as sp  
import numpy as np

x = sp.Symbol('x')
f = x**2
Definite_integral = sp.integrate(f, (x, 0, 2))
indefinite_integral = sp.integrate(f, x)
print("Definite Integral: \n", Definite_integral)
print("Indefinite Integral: \n", indefinite_integral)