import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# m = 100
# X = 6 * np.random.rand(m, 1) - 3
# y = 0.5 * X ** 2 + X + np.random.randn(m, 1)
# poly_feature = PolynomialFeatures(degree = 2, include_bias= False)

# X_poly = poly_feature.fit_transform(X)
# # print(X[0])
# # print(X_poly[0])
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.coef_)
# print(lin_reg.intercept_)
# plt.plot(X, y, 'b.')
# plt.plot(lin_reg.intercept_, lin_reg.coef_, 'r-')
# plt.show()


# X = np.random.randn(300,1)

# X_b = np.c_[np.ones((300, 1)), X]

# y = 4 + 3*X + np.random.randn(300, 1)

# theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# print(theta)

# A = np.array([[0], [2]])

# A_new = np.c_[np.ones((2,1)), A]

# y_predict = A_new.dot(theta)

# plt.plot(X, y, 'b.');
# plt.plot(A, y_predict, "r-");
# plt.axis([0, 2, 0, 15])
# plt.show()

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)

poly_feature = PolynomialFeatures(degree = 2, include_bias= False)

X_poly = poly_feature.fit_transform(X)

X_poly_new = np.c_[np.ones((100,1)),X_poly]

theta = np.linalg.inv(X_poly_new.T.dot(X_poly_new)).dot(X_poly_new.T).dot(y)

print(theta)
P = np.array([[5],[0],[0]])

P_new = np.c_[np.ones((3, 1)), P]
P_new_new = np.c_[np.ones((3,1)),P_new]

y_predict = P_new_new.dot(theta)

plt.plot(X, y, 'b.');
plt.plot(P, y_predict, 'r-');
plt.show()
