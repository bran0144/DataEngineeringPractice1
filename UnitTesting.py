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
# 