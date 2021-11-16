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

# TO run all tests in the tests folder:
# cd tests
# pytest
# this will automatically find tests by recursing through the working dir
# identifies tests by "test_" at beginning of module
# -x flag - stops after first failure
# pytest data/test_preprocessing_helpers.py     only runs tests in that file
# Can select only a test class or unit testthrough its node ID
# <path to test module>::<test class name>
# <path to test module>::<test class name>::<unit test name>
# pytest data/test_preprocessing_helpers.py::TestRowToList
# -k flag - runs tests using keyword expressions
# pytest -k "TestSplitIntoTrainingAndTestingSets"   - will run tests within that test class
# Don't need to use the whole name as long as it is unique
# Can subset with logical operators

# Expected failures
# xfail decorator - marks test as "expected to fail"
class TestTrainModel(object):
    @pytest.mark.xfail(reason="Using TDD, model not implemented")
    def test_on_linear_date(self):
        ...
# Conditional expected failures (like from versioning or other platforms)
# skipif decorator - marks tests to be skipped for certain reasons
class TestConertToInt(object):
    @pytest.mark.skipif(sys.version_info > (2,7))
    def test_with_no_comma(self):
        """Only runs on Python 2.7 or lower"""
        ...
# TO see why a test was skipped:
# -r option (then can add what and where you want to see the reason)
# can also add optional reason argument to xfail
# -s added to -r will print the short reason
# -rx will show the reason for why xfailed tests failed
# decorators can be added to whole classes to avoid having to do it for each test

# Exercises
# Add a reason for the expected failure
@pytest.mark.xfail(reason="Using TDD, model_test() has not yet been implemented")
class TestModelTest(object):
    def test_on_linear_data(self):
        test_input = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        expected = 1.0
        actual = model_test(test_input, 2.0, 1.0)
        message = "model_test({0}) should return {1}, but it actually returned {2}".format(test_input, expected, actual)
        assert actual == pytest.approx(expected), message
        
    def test_on_one_dimensional_array(self):
        test_input = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError) as exc_info:
            model_test(test_input, 1.0, 1.0)

# Import the sys module
import sys

class TestGetDataAsNumpyArray(object):
    # Add a reason for skipping the test
    @pytest.mark.skipif(sys.version_info > (2, 7), reason="Works only on Python 2.7 or lower")
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                             ]
                            )
        actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert actual == pytest.approx(expected), message

# CI and code coverage
# Uses Travis CI
# needs .travis.yml
# sample yml file:
# language: python
# python:
#   - "3.6"
# install:
#   - pip install -e .
#   - pip install pytest-cov codecov
# script:
#   - pytest --cov=src tests
# after_success:
#   - codecov

# Need to install Travis CI through github
# Give app access to necessary repos

# integrates with github and Travis
# TO get the codecov badge:
# update yml with codecov (edited above)
# install codecov app in github
# get the badge from setting in the gui (cut and paste the markdown to the Readme file)

# Using setup and teardown for testing
def test_on_raw_data():
    raw_path, clean_path = raw_and_clean_datafile
    preprocess(raw_data_file_path, clean_data_file_path)
    with open(clean_data_file_path) as f:
        lines = f.readlines()
    first_line = lines[0]
    assert first_line == "1801\t201411\n"
    second_line = lines[1]
    assert second_line == "2002\t333209\n"
    # teardown so that next test gets a clean environment

# In pytest, the setup and teardown happens outside the test in a function called fixture
@pytest.fixture
def my_fixure():
    # Do setup here
    yield data  
    # do teardown

# test accesses the data from fixture by calling fixtured passed as an arg
def test_something(my_fixure):
    ...
    data = my_fixure
    ...
import os
# Setup for fixture for test_on_raw_data:
@pytest.fixure
def raw_and_clean_data_file():
    raw_data_file_path = "raw.txt"
    clean_data_file_path = "clean.txt"
    with open (raw_data_file_path, "w") as f:
        f.write("1,801\t201,411\n"
        "1,767565,122\n"
        "2,002\t33,209\n"
        "1990\t782,911\n"
        "1,285\t389129\n")
    yield raw_data_file_path, clean_data_file_path
    os.remove(raw_data_file_path)
    os.remove(clean_data_file_path)

# tmpdir fixture - builtin
# creates a temp dir during setup and deletes during teardown
# Fixture chaining: setup for tmpdir, setup of raw_and_clean_data_file(), test, teardown of file, teardown of tmpdir
@pytest.fixture
def raw_and_clean_data_file(tmpdir):
   raw_date_file_path = tmpdir.join("raw.txt")
   clean_data_file_path = tmpdir.join("clean.txt")
    #rest of the code stays the same except you don't need to do the os.remove anymore

# Exercises:
# Add a decorator to make this function a fixture
@pytest.fixture
def clean_data_file():
    file_path = "clean_data_file.txt"
    with open(file_path, "w") as f:
        f.write("201\t305671\n7892\t298140\n501\t738293\n")
    yield file_path
    os.remove(file_path)
    
# Pass the correct argument so that the test can use the fixture
def test_on_clean_file(clean_data_file):
    expected = np.array([[201.0, 305671.0], [7892.0, 298140.0], [501.0, 738293.0]])
    # Pass the clean data file path yielded by the fixture as the first argument
    actual = get_data_as_numpy_array(clean_data_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual) 

@pytest.fixture
def empty_file():
    # Assign the file path "empty.txt" to the variable
    file_path = "empty.txt"
    open(file_path, "w").close()
    # Yield the variable file_path
    yield file_path
    # Remove the file in the teardown
    os.remove(file_path)
    
def test_on_empty_file(self, empty_file):
    expected = np.empty((0, 2))
    actual = get_data_as_numpy_array(empty_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)

# Mocking
# Need pytest-mock and unittest.mock (python standard lib)
# mocker.patch("data.prepocessing_helpers.row_to_list")
# returns the MagicMock object which we store in the variable row_to_list_mock
def row_to_list_bug_free(row):
# just creates a dictionary of the inputs and the correct outputs (does not run the function)
    return_values = {...}
    return return_values[row]
# this means that correct values will always be returned
from unittest.mock import call
def test_on_raw_data(raw_and_clean_data_file, mocker,):
    raw_path, clean_path = raw_and_clean_data_file
    row_to_list_mock = mocker.patch("data.preprocessing_helpers.row_to_list")
    row_to_list_mock.side_effect = row_to_list_bug_free

def test_on_raw_data(raw_and_clean_data_file, mocker,):
    raw_path, clean_path = raw_and_clean_data_file
    row_to_list_mock = mocker.patch("data.preprocessing_helpers.row_to_list", side_effect = row_to_list_bug_free)
    preprocess(raw_path, clean_path)
    assert row_to_list_mock.call_args_list == [
        call("1,801\t201,411\n"),
        call("1,767565,112\n")
    ]

row_to_list_mock.call_args_list     #attribute is a list of all the arguments that row_to_list was called with
#args_list is wrapped in a convenience object called cal() that needs to be imported from unittest.mock

# Exercises:

# Define a function convert_to_int_bug_free
def convert_to_int_bug_free(comma_separated_integer_string):
    # Assign to the dictionary holding the correct return values 
    return_values = {"1,801": 1801, "201,411": 201411, "2,002": 2002, "333,209": 333209, "1990": None, "782,911": 782911, "1,285": 1285, "389129": None}
    # Return the correct result using the dictionary return_values
    return return_values[comma_separated_integer_string]

# Add the correct argument to use the mocking fixture in this test
def test_on_raw_data(self, raw_and_clean_data_file, mocker):
    raw_path, clean_path = raw_and_clean_data_file
    # Replace the dependency with the bug-free mock
    convert_to_int_mock = mocker.patch("data.preprocessing_helpers.convert_to_int",
                                       side_effect=convert_to_int_bug_free)
    preprocess(raw_path, clean_path)
    # Check if preprocess() called the dependency correctly
    assert convert_to_int_mock.call_args_list == [call("1,801"), call("201,411"), call("2,002"), call("333,209"), call("1990"), call("782,911"), call("1,285"), call("389129")]
    with open(clean_path, "r") as f:
        lines = f.readlines()
    first_line = lines[0]
    assert first_line == "1801\\t201411\\n"
    second_line = lines[1]
    assert second_line == "2002\\t333209\\n" 

# Testing Models

from data.preprocessing_helpers import preprocess
from features.as_numpy import get_data_as_numpy_array
from models.train import split_into_training_and_testing_sets

preprocess("data/raw/housing_data.txt", "data/clean/clean_housing_data.txt")
data = get_data_as_numpy_array("data/clean/clean_housing_data.txt", 2)
training_set, testing_set = split_into_training_and_testing_sets(data)
slope, intercept = train_model(training_set)


# Linear Regression Model
from scipy.stats import linregress

def train_model(training_set):
    slope, intercept, _,_,_ = linregress(training_set[:, 0], training_set[:,1])
    return slope, intercept

# Hard to test when you don't know the return value (like in a regression model)
# So, one trick is to use a simple dataset where the return value is known
# 
def test_on_linear_data():
    test_argument = np.array([[1.0, 3.0], [2.0, 5.0]])
    expected_slope = 2.0
    expected_intercept = 1.0
    slope, intercept = train_model(test_argument)
    assert slope == pytest.approx(expected_slope)
    assert intercept == pytest.approx(expected_intercept)

# Or we can use inequalities
# If we can't predict the best fit line, but we know the best fit line is positive we can assert it is greater than 0

def test_on_positively_correlated_data():
    test_argument = np.array([[1.0, 4.0], [2.0, 4.0], [3.0, 9.0]])
    slope, intercept = train_model(test_argument)
    assert slope > 0

# Just because models are hard to test, doesn't mean we don't need to
# Try to perform sanity checks

import numpy as np
import pytest
from models.train import model_test

def test_on_perfect_fit():
    # Assign to a NumPy array containing a linear testing set
    test_argument = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    # Fill in with the expected value of r^2 in the case of perfect fit
    expected = 1.0
    # Fill in with the slope and intercept of the model
    actual = model_test(test_argument, slope=2.0, intercept=1.0)
    # Complete the assert statement
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)

def test_on_circular_data(self):
    theta = pi/4.0
    # Assign to a NumPy array holding the circular testing data
    test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                              [0.0, 1.0],
                              [cos(3 * theta), sin(3 * theta)],
                              [-1.0, 0.0],
                              [cos(5 * theta), sin(5 * theta)],
                              [0.0, -1.0],
                              [cos(7 * theta), sin(7 * theta)]]
                             )
    # Fill in with the slope and intercept of the straight line
    actual = model_test(test_argument, slope=0.0, intercept=0.0)
    # Complete the assert statement
    assert actual == pytest.approx(0.0)

#Testing Plots
# Function to test:
#  def get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title):
# returns a matplotlib.figure.Figure() Object
# Has lots of properties including configuration and style, so not advisable to test each individually
# Instead we use one-time baseline generation:
# -decide on test arguments
# -call plotting function on test arguments
# -convert Figure() into PNG image
# -inspect image manually, verify, store as baseline image
# Testing step:
# -call plotting funciton on test arguments
# -convert Figure() to PNG image
# -compare image with stored baseline image
#  Need to use pytest-mpl for image comparisons
# ignores OS related differences
# instead of assert statement, it returns 
# return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)
# Need to add @pytest.mark.mpl_image_compare decorator
# this will do the image comparison and baseline image generation under the hood
# baseline images need to be stored in separate directory in tests/visualization
# to run test:
# pytest -k "test_plot_for_linear_data" --mpl-generate-path visualization/baseline
# next time we run the test we must use the --mpl option to make pyests compare with baseline image
# pytest -k "test_plot_for_linear_data" --mpl
# IF the test fails, both images and the differences by pixel will be saved in a temp folder

# Exercises:
from visualization.plots import get_plot_for_best_fit_line

class TestGetPlotForBestFitLine(object):
    # Add the pytest marker which generates baselines and compares images
    @pytest.mark.mpl_image_compare
    def test_plot_for_almost_linear_data(self):
        slope = 5.0
        intercept = -2.0
        x_array = np.array([1.0, 2.0, 3.0])
        y_array = np.array([3.0, 8.0, 11.0])
        title = "Test plot for almost linear data"
        # Return the matplotlib figure returned by the function under test
        return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)

# command:
# pytest --mpl-generate-path /home/repl/workspace/project/tests/visualization/baseline -k "test_plot_for_almost_linear_data"

