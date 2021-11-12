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

# Testing for failure
import numpy as np
def test_valueerror_on_one_dimenstional_argument():
    example_argument = np.array([2081, 31492, 1059, 186606, 1148, 206186])
    with pytest.raises(ValueError) as exception_info:
        split_into_training_and_testing_sets(example_argument)
    assert exception_info.match("Argument data array must be two dimensional. ")

# If function raises expected ValueError, test will pass. If it is not raised, test will fail
# exception_info stores info about the error (can check after context ends)

# with statements
with context_manager:
    # Runs some code on entering context
    print ("This is part of the context")
    # Runs code on exiting the context
# with takes a single argment called the context manager
with pytest.raises(ValueError):
    # does nothing on entering context
    # If context raised ValueError, it is silenced
    raise ValueError
    # If the context did not raise ValueEror, it raises an exception.
with pytest.raises(ValueError):
    pass    #context exists without raising a Value Error

import numpy as np
import pytest
from train import split_into_training_and_testing_sets

def test_on_one_row():
    test_argument = np.array([[1382.0, 390167.0]])
    # Store information about raised ValueError in exc_info
    with pytest.raises(ValueError) as exc_info:
      split_into_training_and_testing_sets(test_argument)
    expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
    # Check if the raised ValueError contains the correct message
    assert exc_info.match(expected_error_msg)

# Test for argument types
# -Bad arguments - when an exception is raised instead of returning a value
# -Special arguments - boundary values (first value where function works appropriately or after a special logic), 
# when a function uses a special logic to produce the return value (usually a switch or if else)
# -Normal arguments - recommended to test 2-3 normal arguments

# Exercise:
import pytest
from preprocessing_helpers import row_to_list

def test_on_no_tab_no_missing_value():    # (0, 0) boundary value
    # Assign actual to the return value for the argument "123\n"
    actual = row_to_list("123\n")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_two_tabs_no_missing_value():    # (2, 0) boundary value
    actual = row_to_list("123\t4,567\t89\n")
    # Complete the assert statement
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_one_tab_with_missing_value():    # (1, 1) boundary value
    actual = row_to_list("\t4,567\n")
    # Format the failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

def test_on_no_tab_with_missing_value():    # (0, 1) case
    # Assign to the actual return value for the argument "\n"
    actual = row_to_list("\n")
    # Write the assert statement with a failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_two_tabs_with_missing_value():    # (2, 1) case
    # Assign to the actual return value for the argument "123\t\t89\n"
    actual = row_to_list("123\t\t89\n")
    # Write the assert statement with a failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

def test_on_normal_argument_1():
    actual = row_to_list("123\t4,567\n")
    # Fill in with the expected return value for the argument "123\t4,567\n"
    expected = ["123", "4,567"]
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)
    
def test_on_normal_argument_2():
    actual = row_to_list("1,059\t186,606\n")
    expected = ["1,059", "186,606"]
    # Write the assert statement along with a failure message
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)

# Test Driven Development
# Thinking about normal special and bad arguments before implementation makes better code

# How to organize tests
# Test suite should mirror the same directory and package structure
# for each my_module.py there should be a test_my_module.py
# Pytest creates a test class for tests of a specific function

class TestRowToList(object):    #always put the argument object
    def test_on_no_tab_missing_value(self):     #always put the argument self
        ...

class TestCovertToInt(object):
    def test_with_no_comma(self):
        ...

# Declare the test class
class TestSplitIntoTrainingAndTestingSets(object):
    # Fill in with the correct mandatory argument
    def test_on_one_row(self):
        test_argument = np.array([[1382.0, 390167.0]])
        with pytest.raises(ValueError) as exc_info:
            split_into_training_and_testing_sets(test_argument)
        expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
        assert exc_info.match(expected_error_msg)

