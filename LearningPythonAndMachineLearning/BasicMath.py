# a = 12
# b = 24
# addition = a+b
# subtraction = a-b
# multiplication = a * b
# division = a / b
# Max = max(a,b)
# Min = min(a,b)
# print("Addition",addition)
# print("Subtraction",subtraction)
# print("Multiplication",multiplication)
# print("Division",division)
# print("Divided by residual: ",(b % a))
# print("Divided without taking residual: ",(a // b))
# print("Max",Max)
# print("Min",Min)
# print("Abs",abs(-12))
import numpy as np

a = np.array([[0],[2]])
b = np.c_[np.ones((2,1)),a]
print(a)
print(b)