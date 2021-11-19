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

# Object comparison
# if two objects with are compared with == they will not be considered equal (different references)
# BUT, numpy arrays and other objects are compared using data, not their references
# to override the equality:
class Customer:
    def __init(self, id, name):
        self.id, self.name = id, name
    def __eq__(self, other):
        return (self.id == other.id) and (self.name == other.name)
# Now you can use == on the objects
# __hash__() - use objects as dictionary keys in sets
# 

# Exercises:
class BankAccount:
   # MODIFY to initialize a number attribute
    def __init__(self, number, balance=0):
        self.number = number
        self.balance = balance
      
    def withdraw(self, amount):
        self.balance -= amount 
    
    # Define __eq__ that returns True if the number attributes are equal 
    def __eq__(self, other):
        return self.number == other.number  

# Create accounts and compare them       
acct1 = BankAccount(123, 1000)
acct2 = BankAccount(123, 1000)
acct3 = BankAccount(456, 1000)
print(acct1 == acct2)
print(acct1 == acct3)

class BankAccount:
    def __init__(self, number, balance=0):
        self.number, self.balance = number, balance
      
    def withdraw(self, amount):
        self.balance -= amount 

    # MODIFY to add a check for the type()
    def __eq__(self, other):
        return (self.number == other.number) and (type(self) == type(other))

acct = BankAccount(873555333)
pn = Phone(873555333)
print(acct == pn)

# String representation
# __str__() used when we call print - string representation
# __repr__() - reproducible representation - return exact call needed to reproduce the object
def __str__(self):
    cust_str = """Customer: name: {name} balance: {balance}""".format(
        name = self.name, balance = self.balance)
    return cust_str
def __repr__(self):
    return "Customer('{name}', {balance})".format(name = self.name, balance = self.balance)

# Exercises:
class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
            
    # Add the __str__() method
    def __str__(self):
        return 'Employee name: "{name}" \nEmployee salary: {salary}'.format(name=self.name, salary=self.salary)

emp1 = Employee("Amar Howard", 30000)
print(emp1)
emp2 = Employee("Carolyn Ramirez", 35000)
print(emp2)

class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
      

    def __str__(self):
        s = "Employee name: {name}\nEmployee salary: {salary}".format(name=self.name, salary=self.salary)      
        return s
      
    # Add the __repr__method  
    def __repr__(self):
        return """Employee("{name}", {salary})""".format(name=self.name, salary=self.salary)   

emp1 = Employee("Amar Howard", 30000)
print(repr(emp1))
emp2 = Employee("Carolyn Ramirez", 35000)
print(repr(emp2))

# Exceptions
# try, except, finally
# raise - to raise your own excpetions
# Exceptions are classes and can be inherited from to create custom exceptions

class Customer:
    def __init__(self, name, balance):
        if balance < 0:
            raise BalanceError("Balance cannot be negative!")
        else:
            self.name, self.balance = name, balance

# MODIFY the function to catch exceptions
def invert_at_index(x, ind):
    try:
        return 1/x[ind]
    except ZeroDivisionError:
        print("Cannot divide by zero!")
    except IndexError:
        print("Index out of range!")
 
a = [5,6,0,7]

# Works okay
print(invert_at_index(a, 1))

# Potential ZeroDivisionError
print(invert_at_index(a, 2))

# Potential IndexError
print(invert_at_index(a, 5))

class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_RAISE = 5000

  def __init__(self, name, salary = 30000):
    self.name = name
    
    # If salary is too low
    if salary < MIN_SALARY:
      # Raise a SalaryError exception
      raise SalaryError("Salary is too low!")
      
    self.salary = salary

class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_BONUS = 5000

  def __init__(self, name, salary = 30000):
    self.name = name    
    if salary < Employee.MIN_SALARY:
      raise SalaryError("Salary is too low!")      
    self.salary = salary
    
  # Rewrite using exceptions  
  def give_bonus(self, amount):
    if amount > Employee.MAX_BONUS:
       raise BonusError 
        
    elif self.salary + amount <  Employee.MIN_SALARY:
       raise SalaryError
      
    else:  
      self.salary += amount

# Polymorphism
# using a unifed interface to operate on objects of different classes
# Liskov principle: a base class should be interchangeable with any of its subclasses without
# altering any properties of the program
# function signatures are compatible (arguments, return values)
# state of the object and program remain consistent (input and output conditions should reamin the same)
# should not throw additional excpetions

# Controlling access
# name conventions
# starts with _  means internal, not part of public class interface
# obj._att_name
# used for implementation details and helper functions

# __ means "private" (only starts with __ not ends) not inherited, to try to prevent name clashes
# obj.__attr_name

# __attr_name__ are only for built-in python methods

# @property - custom access
# overriding __getattr__() and __setattr__()

# Exercises:
# Add class attributes for max number of days and months
class BetterDate:
    _MAX_DAYS = 30
    _MAX_MONTHS = 12
    
    def __init__(self, year, month, day):
        self.year, self.month, self.day = year, month, day
        
    @classmethod
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
        
    # Add _is_valid() checking day and month values
    def _is_valid(self):
        return (self.day <= BetterDate._MAX_DAYS) and \
               (self.month <= BetterDate._MAX_MONTHS)
        
bd1 = BetterDate(2020, 4, 30)
print(bd1._is_valid())

bd2 = BetterDate(2020, 6, 45)
print(bd2._is_valid())

# Properties
# control attribute access, validity, etc.

class Employer:
    def __init__(self, name, new_salary):
        self._salary = new_salary
    @property
    def salary(self):
        return self._salary
    @salary.setter
    def salary(self, new_salary):
        if new_salary < 0:
           raise ValueError("Invalid Salary")
        self._salary = new_salary 

# Exercise:
# Create a Customer class
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance

    # Add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # Validate the parameter value
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")

# Create a Customer        
cust = Customer("Belinda Lutz", 2000)

# Assign 3000 to the balance property
cust.balance = 3000

# Print the balance property
print(cust.balance)

# MODIFY the class to use _created_at instead of created_at
class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.today()
    
    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self._created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)   
    
    # Add a read-only property: _created_at
    @property  
    def created_at(self):
        return self._created_at

# Instantiate a LoggedDF called ldf
ldf = LoggedDF({"col1": [1,2], "col2":[3,4]}) 

