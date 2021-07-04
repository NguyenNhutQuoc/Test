number  = 1
number_1 = 1000
number_2 = 12.2222
number_3 = "I Love You"
print(type(number))
print(type(number_1))
print(type(number_2))
print(type(number_3))
print(id(number))
print(id(number_1))
print(id(number_2))
print(id(number_3))
# The same ID
print("The Same ID")
ID = 123
ID_1 = 123
print(id(ID))
print(id(ID_1))
# The difference ID
print("The difference ID")
number_5 = "The difference"
ID_2 = "The difference"
# The difference ID belonging to the name
print(id(number_5))
print(id(ID))