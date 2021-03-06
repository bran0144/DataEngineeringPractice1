import numpy as np

# Using range, enumerate, map, and list comprehensions to write more efficient code
# Create a range object that goes from 0 to 5
nums = range(6)
print(type(nums))

# Convert nums to a list
nums_list = list(nums)
print(nums_list)

# Create a new list of odd numbers from 1 to 11 by unpacking a range object
nums_list2 = [*range(1,12,2)]
print(nums_list2)

# Rewrite the for loop to use enumerate
indexed_names = []
for i,name in enumerate(names):
    index_name = (i,name)
    indexed_names.append(index_name) 
print(indexed_names)

# Rewrite the above for loop using list comprehension
indexed_names_comp = [(i , name) for i,name in enumerate(names)]
print(indexed_names_comp)

# Unpack an enumerate object with a starting index of one
indexed_names_unpack = [*enumerate(names, 1)]
print(indexed_names_unpack)

# Use map to apply str.upper to each element in names
names_map  = map(str.upper, names)

# Print the type of the names_map
print(type(names_map))

# Unpack names_map into a list
names_uppercase = [*list(names_map)]

# Print the list created above
print(names_uppercase)

# Numpy, arrays and broadcasting
# arrays allow for boolean indexing
# nums_np = np.array([-2, -1, 0])
# nums_npp ** 2
# num2_np[0,1]
# nums2_np[:,0]
# nums_np > 0
# nums_np[nums_np > 0]

# Print second row of nums
print(nums[1,:])

# Print all elements of nums that are greater than six
print(nums[nums > 6])

# Double every element of nums
nums_dbl = nums * 2
print(nums_dbl)

# Replace the third column of nums
nums[0:,2] = nums[0:,2] + 1
print(nums)

# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]

# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest, guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')

# IPython - magic commands %
# %lsmagic lists all magic commands
# %timeit -r2 -n10 rand_nums = np.random.rand(1000)
# -r (# of runs) -n (number of loops)
# (double %%) %timeit cell magic mode lets you run multiple lines of code
# times = %timeit -o rand_nums = np.random.rand(1000) (saves it into an output)

# Code profiling
# detailed stats on frequency and duration of function calls (summary statistics)
# line_profiler (need to pip install)
# %load_ext line_profiler
# %lprun -f convert_units convert_units(heroes, hts, wts)

# Combining objects with zip
# combined_zip=zip(names, hps)
# returns a zip object that must be unpacked into a list
# combined_zip_list = [*combined_zip]
# Collections module
# namedtuple - tuple subclasses with named fields
# deque - list like container with fast appends and pops
# Counter - dict for counting hashable objects
# OrderedDict - dictionary that retains order of entries
# defaultDict - dict that calls a factory function to supply missing values
# from collections import Counter
# type_counts = Counter(poke_types) poke_types is a list
# returns a dictionary with the counts of each item from list
# itertools module
# itertools.combinations()
# combos_obj= cominations(poke_types, 2) - returns a combinations object
# combos = [*combos_obj]

# Combine all three lists together
names_types = [*zip(names, primary_types, secondary_types)]

print(*names_types[:5], sep='\n')

# Combine five items from names and three items from primary_types
differing_lengths = [*zip(names[:5], primary_types[:3])]

print(*differing_lengths, sep='\n')

# Collect the count of primary types
type_count = Counter(primary_types)
print(type_count, '\n')

# Collect the count of generations
gen_count = Counter(generations)
print(gen_count, '\n')

# Use list comprehension to get each Pok??mon's starting letter
starting_letters = [name[0] for name in names]

# Collect the count of Pok??mon for each starting_letter
starting_letters_count = Counter(starting_letters)
print(starting_letters_count)

# Import combinations from itertools
from itertools import combinations

# Create a combination object with pairs of Pok??mon
combos_obj = combinations(pokemon, 2)
print(type(combos_obj), '\n')

# Convert combos_obj to a list by unpacking
combos_2 = [*combos_obj]
print(combos_2, '\n')

# Collect all possible combinations of 4 Pok??mon directly into a list
combos_4 = [*combinations(pokemon, 4)]
print(combos_4)

# Set theory
# intersection() - all elements that are in both sets
# difference() - all elements in one but not the other
# symmetric_difference() - all elements in exactly one set
# union() - all elements that are in either set
# membership testing (check if a value exists) - use the in operator
# convert each list into a set, then use .intersection() method
# set_a.intersection(set_b)
# Membership testing in a set is much faster than a tuple or list
# all items must be unique in sets

# Convert both lists to sets
ash_set = set(ash_pokedex)
misty_set = set(misty_pokedex)

# Find the Pok??mon that exist in both sets
both = ash_set.intersection(misty_set)
print(both)

# Find the Pok??mon that Ash has and Misty does not have
ash_only = ash_set.difference(misty_set)
print(ash_only)

# Find the Pok??mon that are in only one set (not both)
unique_to_set = ash_set.symmetric_difference(misty_set)
print(unique_to_set)

# Convert Brock's Pok??dex to a set
brock_pokedex_set = set(brock_pokedex)
print(brock_pokedex_set)

# Check if Psyduck is in Ash's list and Brock's set
print('Psyduck' in ash_pokedex)
print('Psyduck' in brock_pokedex_set)

# Check if Machop is in Ash's list and Brock's set
print('Machop' in ash_pokedex)
print('Machop' in brock_pokedex_set)

# Use find_unique_items() to collect unique Pok??mon names
uniq_names_func = find_unique_items(names)
print(len(uniq_names_func))

# Convert the names list to a set to collect unique Pok??mon names
uniq_names_set = set(names)
print(len(uniq_names_set))

# Check that both unique collections are equivalent
print(sorted(uniq_names_func) == sorted(uniq_names_set))

# Use the best approach to collect unique primary types and generations
uniq_types = set(primary_types) 
uniq_gens = set(generations)
print(uniq_types, uniq_gens, sep='\n') 

# Eliminating loops
# flat is better than nested
# using list comprehension: totals_comp = [sum(row) for row in poke_stats]
# using map: totals_map = [*map(sum, poke_stats)]
# numpy array instead of a list of lists
# avgs_np = poke_stats.mean(axis=1)

# Collect Pok??mon that belong to generation 1 or generation 2
gen1_gen2_pokemon = [name for name,gen in zip(poke_names, poke_gens) if gen < 3]

# Create a map object that stores the name lengths
name_lengths_map = map(len, gen1_gen2_pokemon)

# Combine gen1_gen2_pokemon and name_lengths_map into a list
gen1_gen2_name_lengths = [*zip(gen1_gen2_pokemon, name_lengths_map)]

print(gen1_gen2_name_lengths_loop[:5])
print(gen1_gen2_name_lengths[:5])

# Create a total stats array
total_stats_np = stats.sum(axis=1)

# Create an average stats array
avg_stats_np = stats.mean(axis=1)

# Combine names, total_stats_np, and avg_stats_np into a list
poke_list_np = [*zip(names, total_stats_np, avg_stats_np)]

print(poke_list_np == poke_list, '\n')
print(poke_list_np[:3])
print(poke_list[:3], '\n')
top_3 = sorted(poke_list_np, key=lambda x: x[1], reverse=True)[:3]
print('3 strongest Pok??mon:\n{}'.format(top_3))

# Collect all possible pairs using combinations()
possible_pairs = [*combinations(pokemon_types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

# Append each enumerated_pair_tuple to the empty list above
for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)

# Calculate the total HP avg and total HP standard deviation
hp_avg = hps.mean()
hp_std = hps.std()

# Use NumPy to eliminate the previous for loop
z_scores = (hps - hp_avg)/hp_std

# Combine names, hps, and z_scores
poke_zscores2 = [*zip(names, hps, z_scores)]
print(*poke_zscores2[:3], sep='\n')

# Pandas efficiency
# .iloc method may not be very efficient
# iterrows() returns a tuple of the index of each row and the data in each row

# Print the row and type of each row
for row_tuple in pit_df.iterrows():
    print(row_tuple)
    print(type(row_tuple))

# Create an empty list to store run differentials
run_diffs = []

# Write a for loop and collect runs allowed and runs scored for each row
for i,row in giants_df.iterrows():
    runs_scored = row['RS']
    runs_allowed = row['RA']
    
    # Use the provided function to calculate run_diff for each row
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    # Append each run differential to the output list
    run_diffs.append(run_diff)

giants_df['RD'] = run_diffs
print(giants_df)

# itertuples() - often more efficient than iterrows()
# iterrows means that you need to use [] to access values
# row_tuple[1]['Team']
# itertuples() returns a namedtuple
# fields are accessible using attribute lookup (using . notation)
# print(row_namedtuple.Index)
# print(row_namedtuple.Team)

# Loop over the DataFrame and print each row's Index, Year and Wins (W)
for row in rangers_df.itertuples():
  i = row.Index
  year = row.Year
  wins = row.W
  
  # Check if rangers made Playoffs (1 means yes; 0 means no)
  if row.Playoffs == 1:
    print(i, year, wins)

run_diffs = []

# Loop over the DataFrame and calculate each row's run differential
for row in yankees_df.itertuples():
    
    runs_scored = row.RS
    runs_allowed = row.RA

    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    run_diffs.append(run_diff)

# Append new column
yankees_df['RD'] = run_diffs
print(yankees_df)

# pandas .apply() method
# similar to map() it takes a function and applies it to entire DF
# must specify an axis (0 = columns, 1=rows)
# can be used with anonymous functions or lambdas
# baseball_df.apply(lambda row: calc_fun_diff(row['RS'], row['RA']), axis=1)
# can be saved into a variable

# Gather total runs scored in all games per year
total_runs_scored = rays_df[['RS', 'RA']].apply(sum, axis=1)
print(total_runs_scored)

# Convert numeric playoffs to text by applying text_playoffs()
textual_playoffs = rays_df.apply(lambda row: text_playoffs(row['Playoffs']), axis=1)
print(textual_playoffs)

# Display the first five rows of the DataFrame
print(dbacks_df.head())

# Create a win percentage Series 
win_percs = dbacks_df.apply(lambda row: calc_win_perc(row['W'], row['G']), axis=1)
print(win_percs, '\n')

# Append a new column to dbacks_df
dbacks_df['WP'] = win_percs
print(dbacks_df, '\n')

# Display dbacks_df where WP is greater than 0.50
print(dbacks_df[dbacks_df['WP'] >= 0.50])

# can grab values as a numpy array using baseball_df['W'].values
# broadcasting (vectorizing) is extremely efficient
# instead of using .iterrows, itertuples, or .apply, we can use numpy instead
# df['RS'].values - df['RA'].values
# code is also much more readable, no loops

# Use the W array and G array to calculate win percentages
win_percs_np = calc_win_perc(baseball_df['W'].values, baseball_df['G'].values)

# Append a new column to baseball_df that stores all win percentages
baseball_df['WP'] = win_percs_np

print(baseball_df.head())

win_perc_preds_loop = []

# Use a loop and .itertuples() to collect each row's predicted win percentage
for row in baseball_df.itertuples():
    runs_scored = row.RS
    runs_allowed = row.RA
    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)
    win_perc_preds_loop.append(win_perc_pred)

# Apply predict_win_perc to each row of the DataFrame
win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)

# Calculate the win percentage predictions using NumPy arrays
win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)
baseball_df['WP_preds'] = win_perc_preds_np
print(baseball_df.head())