def switcher(choose):
    dicted = {1: "English with ms.Sa"
                ,2:"OOP"
                ,3: "Machine Learning"}
    for i in range(1,len(dicted) + 1):
        print(i,':',dict(dicted)[i])
    return dicted.get(choose,"Invalid")
def yourChoose():
    while True:
        choose = int(input("enter a number: "))
        values = switcher(choose)
        print(values)
        if values == "Invalid":
            break

yourChoose()