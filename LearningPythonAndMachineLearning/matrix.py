from typing import List


def initializeMatrix(m,n):
    matrix = []
    for i in range(m):
        marix_1 =[]
        for j in range(n):
            marix_1.append(int(input("Matrix["+str(i)+"]["+str(j)+"]: ")))
        matrix.append(marix_1)
    return matrix

def getMatrix(matrix):
    print('Output: ')
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j], end='\t')
        print('\n',end='')


def addTwoMatrix(matrix_1, matrix_2):
    matrix = []
    if len(matrix_1) == len(matrix_2) and len(matrix_1[0]) == len(matrix_2[0]):
        for i in range(len(matrix_1)):
            add = []
            for j in range(len(matrix_1[0])):
                add.append(matrix_1[i][j] + matrix_2[i][j])
            matrix.append(add)
    return matrix
def subTwoMatrix(matrix_1, matrix_2):
    matrix = []
    if(len(matrix_1) == len(matrix_2) and len(matrix_1[0]) == len(matrix_2[0])):
        for i in range(len(matrix_1)):
            matrixSub = []
            for j in range(len(matrix_1[0])):
                sub = matrix_1[i][j] - matrix_2[i][j]
                matrixSub.append(sub)
            matrix.append(matrixSub)
    return matrix
def aimlessAccumulation(matrix, k):
    matrixAimlessAccumulation = []
    for i in range(len(matrix)):
        add = []
        for j in range(len(matrix[0])):
            add.append(matrix[i][j])
        matrixAimlessAccumulation.append(add)
    for i in range(len(matrixAimlessAccumulation)):
        for j in range(len(matrixAimlessAccumulation[0])):
            matrixAimlessAccumulation[i][j] *= k
    return matrixAimlessAccumulation
def multiplyTwoMatrices(matrix_1, matrix_2):
    if(len(matrix_1[0]) == len(matrix_2)):
        matrix = []
        for k in range(len(matrix_1)):
            matrixMul = []
            for i in range(len(matrix_1)):
                sum = 0
                for j in range(len(matrix_1[0])):
                    multiply = matrix_1[k][j] * matrix_2[j][i]
                    sum += multiply
                matrixMul.append(sum)
            matrix.append(matrixMul)
        return matrix
    else:
        print('Error')
        return []
m = int(input('Enter a column number: '))
n = int(input('Enter a row number: '))
matrix_1 = initializeMatrix(m,n)
m_1 = int(input('Enter a column number: '))
n_2 = int(input('Enter a row number: '))
matrix_2 = initializeMatrix(m_1,n_2)
print('Add matrix')
getMatrix(addTwoMatrix(matrix_1,matrix_2))
print('Sub matrix')
getMatrix(subTwoMatrix(matrix_1,matrix_2))
print('Aimless accumulations')
getMatrix(aimlessAccumulation(matrix_1,int(input('Enter a number: '))))
print('Multiply two matrix')
getMatrix(multiplyTwoMatrices(matrix_1, matrix_2))