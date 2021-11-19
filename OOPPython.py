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

# Instance level data vs class level data
# Class attributes - are the same for every instance and serve as a global variable in the class
# use all caps and no self to define
class Employee:
    MIN_SALARY = 30000

# Use class name instead of self when referring to it within the class
# Employee.MIN_SALARY
# can also be accessed from an instance
# emp1.MIN_SALARY
# Common use cases:
# min/max values for attributes
# common used values for constants (pi)

# Class methods:
# regular methods are shared, same code for every instance, but data fed in changes
# CLass methods can't use instance data
@classmethod
def new_method(cls, args):
    pass
# can't use instance attributes within the method
# to call a class method:
Employee.new_method(args)
# instead of object.method you use Class.method
# main use case is alternative constructors
# a class can only have one __init__ method
# can use cls(name) to call the __init__ constructor 

# Exercises:
# Create a Player class
class Player:
    MAX_POSITION = 10
    def __init__(self, position=0):
        self.position = position


# Print Player.MAX_POSITION       
print(Player.MAX_POSITION)

# Create a player p and print its MAX_POSITITON
p = Player()
print(p.MAX_POSITION)

class Player:
    MAX_POSITION = 10
    
    def __init__(self):
        self.position = 0

    # Add a move() method with steps parameter
    def move(self, steps):
        if self.position + steps < Player.MAX_POSITION:
            self.position = self.position + steps
        else:
            self.position = Player.MAX_POSITION
      
    # This method provides a rudimentary visualization in the console    
    def draw(self):
        drawing = "-" * self.position + "|" +"-"*(Player.MAX_POSITION - self.position)
        print(drawing)

p = Player(); p.draw()
p.move(4); p.draw()
p.move(5); p.draw()
p.move(3); p.draw()

# Create Players p1 and p2
p1 = Player()
p2 = Player()

print("MAX_SPEED of p1 and p2 before assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

# Assign 7 to p1.MAX_SPEED
p1.MAX_SPEED = 7

print("MAX_SPEED of p1 and p2 after assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

print("MAX_SPEED of Player:")
# Print Player.MAX_SPEED
print(Player.MAX_SPEED)

# Create Players p1 and p2
p1, p2 = Player(), Player()

print("MAX_SPEED of p1 and p2 before assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

# ---MODIFY THIS LINE--- 
Player.MAX_SPEED = 7

print("MAX_SPEED of p1 and p2 after assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

print("MAX_SPEED of Player:")
# Print Player.MAX_SPEED
print(Player.MAX_SPEED)

class BetterDate:    
    # Constructor
    def __init__(self, year, month, day):
      # Recall that Python allows multiple variable assignments in one line
      self.year, self.month, self.day = year, month, day
    
    # Define a class method from_str
    @classmethod
    def from_str(cls, datestr):
        # Split the string at "-" and convert each part to integer
        parts = datestr.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # Return the class instance
        return BetterDate(year, month, day)
        
bd = BetterDate.from_str('2020-04-30')   
print(bd.year)
print(bd.month)
print(bd.day)

# import datetime from datetime
from datetime import datetime

class BetterDate:
    def __init__(self, year, month, day):
      self.year, self.month, self.day = year, month, day
      
    @classmethod
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
      
    # Define a class method from_datetime accepting a datetime object
    @classmethod
    def from_datetime(cls, dateobj):
      year, month, day = dateobj.year, dateobj.month, dateobj.day
      return cls(year, month, day) 


# You should be able to run the code below with no errors: 
today = datetime.today()     
bd = BetterDate.from_datetime(today)   
print(bd.year)
print(bd.month)
print(bd.day)

# Inheritance - is-a relationship
# Can customize old classes to make new classes so that you can add new attributes and methods
# can reuse lots of code so that you aren't redoing everything

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    def withdraw(self, amount):
        self.balance -= amount

class SavingsAccount(BankAccount):
    def __init__(self, balance, interest_rate):
        # they use this way BankAccount.__init__(self, balance)
        super().__init__(balance)
        self.interest_rate = interest_rate

class CheckingAccount(BankAccount):
    def __init__(self, balance, limit):
        # they use this way BankAccount.__init__(self, content)
        super().__init__(balance)
        self.limit = limit
    def deposit(self, amount):
        self.balance += amount
    def withdraw(self, amount, fee=0):
        if fee <= self.limit:
            BankAccount.withdraw(self, amount - fee)
        else:
            BankAccount.withdraw(self, amount - self.limit)

class Employee:
  MIN_SALARY = 30000    

  def __init__(self, name, salary=MIN_SALARY):
      self.name = name
      if salary >= Employee.MIN_SALARY:
        self.salary = salary
      else:
        self.salary = Employee.MIN_SALARY
        
  def give_raise(self, amount):
      self.salary += amount      
        
# Define a new class Manager inheriting from Employee
class Manager(Employee):
  pass

# Define a Manager object
mng = Manager("Debbie Lashko", 86500)

# Print mng's name
print(mng.name)

class Employee:
  MIN_SALARY = 30000    

  def __init__(self, name, salary=MIN_SALARY):
      self.name = name
      if salary >= Employee.MIN_SALARY:
        self.salary = salary
      else:
        self.salary = Employee.MIN_SALARY
  def give_raise(self, amount):
    self.salary += amount      
        
# MODIFY Manager class and add a display method
class Manager(Employee):
  def display(self):
    print("Manager", self.name)

mng = Manager("Debbie Lashko", 86500)
print(mng.name)

# Call mng.display()
mng.display()

class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
  # Add a constructor 
    def __init__(self, name, salary=50000, project=None):

        # Call the parent's constructor   
        Employee.__init__(self, name, salary)

        # Assign project attribute
        self.project = project  

  
    def display(self):
        print("Manager ", self.name)
 
 class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
    def display(self):
        print("Manager ", self.name)

    def __init__(self, name, salary=50000, project=None):
        Employee.__init__(self, name, salary)
        self.project = project

    # Add a give_raise method
    def give_raise(self, amount, bonus=1.05):
        Employee.give_raise(self, amount * bonus)
    
    
mngr = Manager("Ashta Dunbar", 78500)
mngr.give_raise(1000)
print(mngr.salary)
mngr.give_raise(2000, bonus=1.03)
print(mngr.salary)

# Define LoggedDF inherited from pd.DataFrame and add the constructor
class LoggedDF(pd.DataFrame):
  
  def __init__(self, *args, **kwargs):
    pd.DataFrame.__init__(self, *args, **kwargs)
    self.created_at = datetime.today()
    
  def to_csv(self, *args, **kwargs):
    # Copy self to a temporary DataFrame
    temp = self.copy()
    
    # Create a new column filled with self.created_at
    temp["created_at"] = self.created_at
    
    # Call pd.DataFrame.to_csv on temp, passing in *args and **kwargs
    pd.DataFrame.to_csv(temp, *args, **kwargs)

    