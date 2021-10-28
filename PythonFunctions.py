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
     