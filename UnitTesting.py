# Some python libraries used for testing: pytest, unittest, nosetests, doctest
# Tests should have their own .py file and should start with test_
# Each written test module should start with test_ and describe what they are testing
# To run the tests: (in command line)
# pytest test_row_to_list.py
# to run command line expression in Datacamp console use ! before the expression
# running test with output test report

# Example:
# Import the pytest package
import pytest

# Import the function convert_to_int()
from preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
  # Complete the assert statement
  assert convert_to_int("2,081") == 2081

# Test result report:
# Section 1: General information (OS, python version, pytest version, rootdir, and plugins)
# Section 2: "Collected 3 items" -means pytest found 3 tests to run
# F - failure - an exception is raised when running unit test
# . - means unit test passed
# Section 3: detailed information about failed tests
# line raising the error has a > 
# Section 4: Test summary, lists number of tests that passed and failed
# also lists running time

# Tests can also serve as documentation
# !cat test_row_to_list.py
# Can increase trust that functions are working
# Unit tests in CI - runs the tests before pushing to ensure only good code is pushed
# 
# Unit tests are small, independent - test one function or class
# Integration tests check if multiple units work
# End to end tests - check the whole software at once

# Assert statements
# assert boolean, message
# message is only printed when an AssertionError was raised
# 
# Beward of float return values!
# Because of how floats are calculated (out to very small decimals), we can't use == to compare
# pytest.approx() to wrap expected return value
# assert 0.1 + 0.1 + 0.1 == pytest.approx(0.3)
# also works with numpy arrays containing floats
# can have more than one assert statement

def test_on_string_with_one_comma():
    return_value = convert_to_int("2,081")
    assert isinstance(return_value, int)
    assert return_value == 2081

# new test will only pass if both assert statements are True

# Exercises:
def test_on_string_with_one_comma():
    test_argument = "2,081"
    expected = 2081
    actual = convert_to_int(test_argument)
    # Format the string with the actual return value
    message = "convert_to_int('2,081') should return the int 2081, but it actually returned {0}".format(actual)
    # Write the assert statement which prints message on failure
    assert actual is expected, message

def test_on_clean_file():
  expected = np.array([[2081.0, 314942.0],
                       [1059.0, 186606.0],
  					   [1148.0, 206186.0]
                       ]
                      )
  actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
  message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
  # Complete the assert statement
  assert actual == pytest.approx(expected), message

def test_on_six_rows():
    example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                 [1148.0, 206186.0], [1506.0, 248419.0],
                                 [1210.0, 214114.0], [1697.0, 277794.0]]
                                )
    # Fill in with training array's expected number of rows
    expected_training_array_num_rows = 4
    # Fill in with testing array's expected number of rows
    expected_testing_array_num_rows = 2
    actual = split_into_training_and_testing_sets(example_argument)
    # Write the assert statement checking training array's number of rows
    assert actual[0].shape[0] == expected_training_array_num_rows, "The actual number of rows in the training array is not {}".format(expected_training_array_num_rows)
    # Write the assert statement checking testing array's number of rows
    assert actual[1].shape[1] == expected_testing_array_num_rows, "The actual number of rows in the testing array is not {}".format(expected_testing_array_num_rows)

  