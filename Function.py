def sumOfProduct(lst1, lst2):
    '''
    This function takes two list and returns sum of the product
    '''
    lstsave = []
    for i in range(len(lst1)):
        x = lst1[i]*lst2[i]
        lstsave.append(x)

    z = 0
    for i in range(len(lstsave)):
        z = lstsave[i]+ z

    return z

'''Checking the Function'''
sumOfProduct([2,8],[5,6])

'''Use of lambda'''
x = lambda a, b : a*b
x(2,3)

'''Some more Funtions'''
def myfunc(n):
  return lambda a : a * n

mytripler = myfunc(3)
mydouble = myfunc(2)
print(mytripler(11))
print(mydouble(4))


def wages(x):
    if x >= 100:
        x = x ** 2
        return x
    else:
        return "save more!"

wages(120)

def FUNCTION_TEN(z):
    a = z **5
    return a

FUNCTION_TEN(3)

def wage(x):
    return x * 35
def wage_bonus(y):
    return wage(y)* 2

wage(6),wage_bonus(6)


def savings(d):
    if d >= 100:
        d = d + 20
        return d
    else:
        return " save more"

savings(80)



def family(lastname, firstname):
    return lastname,firstname

family(lastname = "ram",firstname ="swappy")



def infor_triangle(a,b):
    z = a * 3
    d = b * 4
    return (z, d)

infor_triangle(3, 6)


def bellos(x):
    return x - 2
def callos(y):
    return bellos(y)

bellos(5)


def car(numbers):
    total = 0
    for x in numbers:
        if x < 20:
            total += 1
    return total











