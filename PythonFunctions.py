# Docstrings - use triple """"
# Description of function
# Description of arguments
# Description of return values
# Description of errors raised
# Possbile examples of use
# Google style
""" Description of what function does.

    Args: 
        arg_1 (str): Description of arg1
        arg_2 (int, optional): Description of arg2 

    Returns:
        bool: Optional description of return value

    Raises:
        ValueError: 

    Notes:
        See https..... for more info
    """
# Numpy style
""" 
Description of what function does.

Parameters
----------
arg_1 : expected type of arg1
    Description of arg_1
arg_2 : int, optional
    Optional value has a default value.
    Default = 42

Returns
-------
The type of the return value
    Can include description of return value.

"""
# __doc__ attribute holds docstring information
# inspect module has getdoc() that returns a cleaner version wihtout white space
# 

# Get the "count_letter" docstring by using an attribute of the function
docstring = count_letter.__doc__

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))

def standardize(column):
  """Standardize the values in a column.

  Args:
    column (pandas Series): The data to standardize.

  Returns:
    pandas Series: the values as z-scores
  """
  # Finish the function so that it returns the z-scores
  z_score = (column - column.mean()) / column.std()
  return z_score

# Use the standardize() function to calculate the z-scores
df['y1_z'] = standardize(df.y1_gpa)
df['y2_z'] = standardize(df.y2_gpa)
df['y3_z'] = standardize(df.y3_gpa)
df['y4_z'] = standardize(df.y4_gpa)

# Use an immutable variable for the default argument
def better_add_column(values, df=None):
  """Add a column of `values` to a DataFrame `df`.
  The column will be named "col_<n>" where "n" is
  the numerical index of the column.

  Args:
    values (iterable): The values of the new column
    df (DataFrame, optional): The DataFrame to update.
      If no DataFrame is passed, one is created by default.

  Returns:
    DataFrame
  """
  # Update the function to create a default DataFrame
  if df is None:
    df = pandas.DataFrame()
  df['col_{}'.format(len(df.columns))] = values
  return df

# Context manager: sets up and then takes down a context
# with <context-manager>(args) as variable_name:
#   indented (compound statement) code gets run here
# non-indented - outside of context

# Open "alice.txt" and assign the file to "file"
with open('alice.txt') as file:
  text = file.read()

n = 0
for word in text.split():
  if word.lower() in ['cat', 'cats']:
    n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))

image = get_image_from_instagram()

# Time how long process_with_numpy(image) takes to run
with timer():
  print('Numpy version')
  process_with_numpy(image)

# Time how long process_with_pytorch(image) takes to run
with timer():
  print('Pytorch version')
  process_with_pytorch(image)

# Creating a context manager
# @contextlib.contextmanager  (must have this decorator)
# def my_context():
#   add any set up code (optional)
#   yield (going to give a return value, but the function will continue)
#   add any teardown code (optional)
# very handy for reading from a db (hides connecting and disconnecting db)

# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.

  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))

with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)

@contextlib.contextmanager
def open_read_only(filename):
  """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
  read_only_file = open(filename, mode='r')
  # Yield read_only_file so it can be assigned to my_file
  yield read_only_file
  # Close read_only_file
  read_only_file.close()

with open_read_only('my_file.txt') as my_file:
  print(my_file.read())

# Can use for loops with contexts
# And can use nexted contexts
# with open('file1.txt') as file1:
#   with open('file2.txt') as file2: 
#       for line in file2
#           file2.write(line)
# 
# Error statements:
# try:
#   code that might raise an error
# except:
#   handle the error
# finally:
#   this code runs no matter what

# Use the "stock('NVDA')" context manager
# and assign the result to the variable "nvda"
with stock('NVDA') as nvda:
  # Open "NVDA.txt" for writing as f_out
  with open('NVDA.txt', 'w') as f_out:
    for _ in range(10):
      value = nvda.price()
      print('Logging ${:.2f} for NVDA'.format(value))
      f_out.write('{:.2f}\n'.format(value))

def in_dir(directory):
  """Change current working directory to `directory`,
  allow the user to run some code, and change back.

  Args:
    directory (str): The path to a directory to work in.
  """
  current_dir = os.getcwd()
  os.chdir(directory)

  # Add code that lets you handle errors
  try:
    yield
  # Ensure the directory is reset,
  # whether there was an error or not
  finally:
    os.chdir(current_dir)

def in_dir(directory):
  """Change current working directory to `directory`,
  allow the user to run some code, and change back.

  Args:
    directory (str): The path to a directory to work in.
  """
  current_dir = os.getcwd()
  os.chdir(directory)

  # Add code that lets you handle errors
  try:
    yield
  # Ensure the directory is reset,
  # whether there was an error or not
  finally:
    os.chdir(current_dir)

def create_math_function(func_name):
  if func_name == 'add':
    def add(a, b):
      return a + b
    return add
  elif func_name == 'subtract':
    # Define the subtract() function
    def subtract(arg1, arg2):
      return arg1 - arg2
    return subtract
  else:
    print("I don't know that one")
    
add = create_math_function('add')
print('5 + 2 = {}'.format(add(5, 2)))

subtract = create_math_function('subtract')
print('5 - 2 = {}'.format(subtract(5, 2)))

call_count = 0

def my_function():
  # Use a keyword that lets us update call_count 
  global call_count
  call_count += 1
  
  print("You've called my_function() {} times!".format(
    call_count
  ))
  
for _ in range(20):
  my_function()

def return_a_func(arg1, arg2):
  def new_func():
    print('arg1 was {}'.format(arg1))
    print('arg2 was {}'.format(arg2))
  return new_func
    
my_func = return_a_func(2, 17)

# Show that my_func()'s closure is not None
print(my_func.__closure__ is not None)

def my_special_function():
  print('You are running my_special_function()')
  
def get_new_func(func):
  def call_func():
    func()
  return call_func

new_func = get_new_func(my_special_function)

# Redefine my_special_function() to just print "hello"
def my_special_function():
  print("hello")

new_func()

# Using decorators and nested functions
def my_function(a, b, c):
  print(a + b + c)

# Decorate my_function() with the print_args() decorator
my_function = print_args(my_function)

my_function(1, 2, 3)

# Decorate my_function() with the print_args() decorator
@print_args
def my_function(a, b, c):
  print(a + b + c)

my_function(1, 2, 3)

def print_before_and_after(func):
  def wrapper(*args):
    print('Before {}'.format(func.__name__))
    # Call the function being decorated with *args
    func(*args)
    print('After {}'.format(func.__name__))
  # Return the nested function
  return wrapper

@print_before_and_after
def multiply(a, b):
  print(a * b)

multiply(5, 10)

# Decorator functions
# def timer(func):
#   def wrapper(*args, **kwargs):
#       t_start = time.time()
#       result = func(*args, **kwargs)
#       t_total = time.time() - t_start
#       print('{} took {}s'.format(func.__name__, t_total))
#       return result
#   return wrapper
# 
# @timer
# def sleep_in_seconds(n):
#   time.sleep(n)
# 
# def memoize(func):
#   cache = {}
#   def wrapper():
#       if (args, kwargs) not in cache:
#           cache[(args, kwargs)] = func(*args, **kwargs)
#       return cache[(args, kwargs)]
#   return wrapper

# @memoize
# def slow_function(a,b):
#   print('Sleeping...')
#   time.sleep(5)
#   return a + b

def print_return_type(func):
  # Define wrapper(), the decorated function
  def wrapper(*args, **kwargs):
    # Call the function being decorated
    result = func(*args, **kwargs)
    print('{}() returned type {}'.format(
      func.__name__, type(result)
    ))
    return result
  # Return the decorated function
  return wrapper
  
@print_return_type
def foo(value):
  return value
  
print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))

def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    # Call the function being decorated and return the result
    return func
  wrapper.count = 0
  # Return the new decorated function
  return wrapper

# Decorate foo() with the counter() decorator
@counter
def foo():
  print('calling foo()')
  
foo()
foo()

print('foo() was called {} times.'.format(foo.count))

# docstrings and metadata for nested functions will not show up without wraps
# use from functools import wraps
# and use @wraps(func) as a decorator for the wrapper in the wrapper
# function

from functools import wraps

def add_hello(func):
  # Decorate wrapper() so that it keeps func()'s metadata
  @wraps(func)
  def wrapper(*args, **kwargs):
    """Print 'hello' and then call the decorated function."""
    print('Hello')
    return func(*args, **kwargs)
  return wrapper
  
@add_hello
def print_sum(a, b):
  """Adds two numbers and prints the sum"""
  print(a + b)
  
print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)

@check_everything
def duplicate(my_list):
  """Return a new list that repeats the input twice"""
  return my_list + my_list

t_start = time.time()
duplicated_list = duplicate(list(range(50)))
t_end = time.time()
decorated_time = t_end - t_start

t_start = time.time()
# Call the original function instead of the decorated one
duplicated_list = duplicate.__wrapped__(list(range(50)))
t_end = time.time()
undecorated_time = t_end - t_start

print('Decorated time: {:.5f}s'.format(decorated_time))
print('Undecorated time: {:.5f}s'.format(undecorated_time))

# Adding arguments to decorators
# decorators are only suppoosed to take one argument (func)
# Need to make it a function that returns a decorator, but isn't a decorator itself
# def run_n_times(n):
#   def decorator(func):
#       def wrapper(*args, **kwargs):
#           for i in range(n):
#               func(*args, kwargs)
#       return wrapper
#   return decorator
# 
# can then use @run_n_times(3)  to give it an argument
# 

def html(open_tag, close_tag):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      msg = func(*args, **kwargs)
      return '{}{}{}'.format(open_tag, msg, close_tag)
    # Return the decorated function
    return wrapper
  # Return the decorator
  return decorator

# Make hello() return bolded text
@html('<b>', '</b>')
def hello(name):
  return 'Hello {}!'.format(name)
  
print(hello('Alice'))

# Timeout decorator
# import signal
# def raise_timeout(*args, **kwargs):
#   raise TimeoutError()
# signal.signal(signalnum=signal.SIGALRM, handler=raise_timeout)
# signal.alarm(5)
# signal.alarm(0)

# def timeout_in_5s(func):
#   @wraps(func)
#   def wrapper(*args, **kwargs):
#       signal.alarm(5)
#       try:
#           return func(*args, **kwargs)
#       finally:
#           signal.alarm(0)
#   return wrapper

# def timeout(n_seconds):
#   def decorator(func):
#       @wraps(func)
#       def wrapper(*args, **kwargs):
#           signal.alarm(n_seconds)
#           try:
#               return func(*args, **kwargs)
#           finally:
#               singal.alarm(0)
#       return wrapper
#   return decorator

