# Procedural Programming :
# - code as a sequence of steps
# - great for data analysis and scripts

# OOP:
# - code as interactions of objects
# - great for building frameworks and tools
#  - lets you organize your code better, make it more reusable and maintainable
#  - Objects incorporate state and behavior
# - state and behavior are under one unit representing the object (encapsulation)

# Classes:
# - like blueprints for objects
# - describe the possible states and behaviors the object could have

# Python:
# - everything is an object (strings, methods, functions)
# - every object has a class
# - type() - lets you find the class
# - state = attributes - represented by variables = obj.attribute
# - behavior = methods - represented by functions = obj.method()
# dir() - lists all the attributes and methods an object has

# Creating classes:
class Customer:
#    pass #makes an empty class 
    # old version
    # def identify(self, name):       #every method must have self as the first argument
    #     print("I am Custerome " + name)
    def set_name(self, new_name):
        self.name = new_name
    def identify(self):
        print("I am Customer" + self.name)


# Creating an Object
cust = Customer()
cust.set_name("John Smith")

# Exercise:

class Employee:
    def set_name(self, new_name):
        self.name = new_name

    def set_salary(self, new_salary):
        self.salary = new_salary 

    def give_raise(self, amount):
        self.salary = self.salary + amount

    # Add monthly_salary method that returns 1/12th of salary attribute
    def monthly_salary(self):
        return self.salary / 12

    
emp = Employee()
emp.set_name('Korel Rossi')
emp.set_salary(50000)

# Get monthly salary of emp and assign to mon_sal
mon_sal = emp.monthly_salary()

# Print mon_sal
print(mon_sal)

# __init__ constructor
class Customer:
    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance

cust = Customer("John Smith", 1000)
print(cust.name)
print (cust.balance)

# Can set default values for arguments
# We can define attributes in methods (and then calling them will add the attribute to the object) 
# or in constructors where they are set all together (much quicker and better code)
# Best practices:
# initialize attributes in __init__
# Use CamelCase for classes, lower_snake_case for functions and attributes
# Keep self as self
# Use docstrings

# Exercises:
# Import datetime from datetime
from datetime import datetime

class Employee:
    
    def __init__(self, name, salary=0):
        self.name = name
        if salary > 0:
          self.salary = salary
        else:
          self.salary = 0
          print("Invalid salary!")
          
        # Add the hire_date attribute and set it to today's date
        self.hire_date = datetime.today()
        
   # ...Other methods omitted for brevity ...
      
emp = Employee("Korel Rossi", -1000)
print(emp.name)
print(emp.salary)

# Write the class Point as outlined in the instructions
from math import sqrt

class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y 
    def distance_to_origin(self):
        return sqrt((self.x ** 2) + (self.y ** 2))

    def reflect(self, axis):
        if axis == "x":
            self.y = -self.y 
        elif axis == "y":
            self.x = -self.x 
        else:
            print("Error")


pt = Point(x=3.0)
pt.reflect("y")
print((pt.x, pt.y))
pt.y = 4.0
print(pt.distance_to_origin())