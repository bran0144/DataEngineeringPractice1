from typing_extensions import TypeAlias
import pandas as pd
import matplotlib as plt

tax_data = pd.read_csv("us_tax_data_2016.tsv", sep="\t")

# Load TSV using the sep keyword argument to set delimiter
data = pd.read_csv("vt_tax_data_2016.tsv", sep="\t")

# Plot the total number of tax returns by income group
counts = data.groupby("agi_stub").N1.sum()
counts.plot.bar()
plt.show()

# Create list of columns to use
cols = ['zipcode', 'agi_stub', 'mars1', 'MARS2', 'NUMDEP']

# Create data frame from csv using only selected columns
data = pd.read_csv("vt_tax_data_2016.csv", usecols=cols)

# View counts of dependents and tax returns by income level
print(data.groupby("agi_stub").sum())

# Create data frame of next 500 rows with labeled columns
vt_data_next500 = pd.read_csv("vt_tax_data_2016.csv", 
                       		  nrows=500,
                       		  skiprows=500,
                       		  header=None,
                       		  names=list(vt_data_first500))

# View the Vermont data frames to confirm they're different
print(vt_data_first500.head())
print(vt_data_next500.head())

# dtype - takes a dictionary of column names and data Types
# na_values - takes a single value, list, or dictionary of columns and values
# error_bad_lines = False - skips unparseable records
# warn_bad_lines = True - see messages when records are skipped

# Create dict specifying data types for agi_stub and zipcode
data_types = {"agi_stub":"category",
			  "zipcode":"object"}

# Load csv using dtype to set correct data types
data = pd.read_csv("vt_tax_data_2016.csv", dtype=data_types)

# Print data types of resulting frame
print(data.dtypes.head())

# Create dict specifying that 0s in zipcode are NA values
null_values = {"zipcode":0}

# Load csv using na_values keyword argument
data = pd.read_csv("vt_tax_data_2016.csv", 
                   na_values=null_values)

# View rows with NA ZIP codes
print(data[data.zipcode.isna()])

try:
  # Set warn_bad_lines to issue warnings about bad records
  data = pd.read_csv("vt_tax_data_2016_corrupt.csv", 
                     error_bad_lines=False, 
                     warn_bad_lines=True)
  
  # View first 5 records
  print(data.head())
  
except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")

#  read_excel()
# nrows, skiprows, and usecols works similarly
# Load pandas as pd
import pandas as pd

# Read spreadsheet and assign it to survey_responses
survey_responses = pd.read_excel("fcc_survey.xlsx")

# View the head of the data frame
print(survey_responses.head())

# Create string of lettered columns to load
col_string = "AD, AW:BA"

# Load data with skiprows and usecols set
survey_responses = pd.read_excel("fcc_survey_headers.xlsx", 
                        skiprows=2, 
                        usecols=col_string)

# View the names of the columns selected
print(survey_responses.columns)

# sheet_name  - keyword arg loads other sheets in a workbook (default is only 1st)
# can use name, posiiton number (0 indexed), list of sheets

all_responses = pd.DataFrame()

for sheet_name, frame in survey_responses.items():
    frame["Year"] = sheet_name
    all_responses = all_responses.append(frame)
print(all_responses.Year.unique())

# Create df from second worksheet by referencing its name
responses_2017 = pd.read_excel("fcc_survey.xlsx",
                               sheet_name="2017")

# Graph where people would like to get a developer job
job_prefs = responses_2017.groupby("JobPref").JobPref.count()
job_prefs.plot.barh()
plt.show()

# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name=None)

# View the sheet names in all_survey_data
print(all_survey_data.keys())

# Create an empty data frame
all_responses = pd.DataFrame()

# Set up for loop to iterate through values in responses
for df in responses.values():
  # Print the number of rows being added
  print("Adding {} rows".format(df.shape[0]))
  # Append df to all_responses, assign result
  all_responses = all_responses.append(df)

# Graph employment statuses in sample
counts = all_responses.groupby("EmploymentStatus").EmploymentStatus.count()
counts.plot.barh()
plt.show()