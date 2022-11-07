# Example: How long does it take for an investment to reach a target amount?
# Input a deposit amount, a target amount, and an annual interest rate.
# Calculate how many years it takes to reach the target.
deposit = float(input("Enter deposit: "))
target = float(input('Enter target: '))
interest_rate = float(input('Enter interest rate(like .09 for 9 percent): '))
years = 0

balance = deposit

while balance <= target:
    balance = balance * (1 + interest_rate)
    #years = years + 1
    years += 1
    print(years, balance)

print(years)


'''For Range'''
# Example: soy bean quantity and price
quantity = float(input("Start quantity: "))
year = int(input("Start year: "))

# print the header for the output table
print("Year", "Quantity", 'Price')

# complete the code below
for num in range(12):
    price = 20 - (0.1 * quantity)
    # This year's numbers are ready for output.
    print(year, '  ' , format(quantity, '.1f'), format(price, '.2f'))

    # increment the year and calculate next year's quantity
    year = year + 1
    quantity = (5 * price) - 10


# Example: Add input validation to our earlier example of calculating how many years it
#          takes to get to an investment target.
# First, copy your working code from the first example cell. Then we will modify it.
# Example: How long does it take for an investment to reach a target amount?
# Input a deposit amount, a target amount, and an annual interest rate.
# Calculate how many years it takes to reach the target.
deposit = float(input("Enter deposit: "))
while deposit < 1:
    deposit = float(input("Your entry is invalid. Enter deposit: "))

target = float(input('Enter target: '))


interest_rate = float(input('Enter interest rate as a decimal: '))
years = 0

balance = deposit

while not(balance > target):
    balance = balance * (1 + interest_rate)
    #years = years + 1
    years += 1
    print(years, balance)

print(years)
