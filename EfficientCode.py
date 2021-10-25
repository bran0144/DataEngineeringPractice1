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

