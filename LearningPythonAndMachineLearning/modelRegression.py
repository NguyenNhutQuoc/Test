import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import pandas as pd


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=",",delimiter = "\t",encoding = "latin1",na_values= "n/a")

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
# X = np.c_[country_stats["GDP per capita"]]
# y = np.c_[country_stats["Life satisfaction"]]
country_stats.plot(kind = 'scatter', x = 'GDP per capita', y = 'Life satisfaction')
# plt.show()

X = 2 * np.random.rand(100,1)
y = 4 + 3* X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
# # use theta
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
# print(y_predict)
# plt.plot(X_new, y_predict, "r-")
# plt.plot(X,y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()
#Use model
# from sklearn.linear_model import LinearRegression

# lin_reg = LinearRegression()

# lin_reg.fit(X, y)
# print(lin_reg.predict(X_new))

# # using the Gradient Descent step

# eta = 0.1

# n_iteration = 1000

# m = 100

# theta = np.random.randn(2,1)

# for iteration in range(n_iteration):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients

# print(theta)
# #using the learning_schedule
# n_epochs = 50

# t0, t1 = 5, 50

# def learning_schedule(t):
#     return t0 / (t + t1)

# theta = np.random.randn(2, 1)

# for epcho in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index: random_index + 1]
#         yi = y[random_index: random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epcho * m + i)
#         theta = theta - eta * gradients

# print(theta)

# # using SGDRegressor  in sklearn.linear_model

# from sklearn.linear_model import SGDRegressor

# sgd_regressor = SGDRegressor(penalty = None)

# sgd_regressor.fit(X, y.ravel())

# print(sgd_regressor.intercept_, sgd_regressor.coef_)

#Polynomial regression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)

# plt.plot(X, y, 'b.')
# plt.show()