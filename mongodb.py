#JSON - JavaScript Object Notation
#2 collection structures
    #Objects - maps string keys and values - order is not important - similar to dict
    #Arrays - series of values, order of values is important - similar to lists
#null in JSON is like None in Python
#In decreasing size (DB holds collections, which holds documents, which holds fields)
#MongoDB            JSON                Python
#Databases          Objects             Dictionaries
#COllections        Arrays              Lists
#Documents          Objects             Dictionaries
#Subdocuments       Objects             Dictionaries
#Fields/Values      Value Types         Types (including datetime and regex)

# Using Nobel Prize API database
import requests
from pymongo import MongoClient

client = MongoClient()
db = client["nobel"]
for collection_name in ["prizes", "laureates"]:
    response = requests.get("http://api.nobelprize.org/v1/{}.json".format(collection_name[:-1]))
    documents = response.json()[collection_name]
    db[collection_name].insert_many(documents)

#Accessing db and collections Using []
#client is a dictionary of databases
db = client["nobel"]
#database is a dictionary of collections
prizes_collection = db["prizes"]

#Using .
#databases are attributes of a client
db = client.nobel
#collections are attributes of db's
prizes_collection = db["prizes"]

#Count documents in a collection
#use empty document {} as a filter
filter = {}
#Count documents in a collection
n_prizes = db.prizes.count_documents(filter)
n_laureates = db.laureates.count_documents(filter)

#Find one document to inspect
doc = db.prizes.find_one(filter)

#Exercises:
# Save a list of names of the databases managed by client
db_names = client.list_database_names()
print(db_names)

# Save a list of names of the collections managed by the "nobel" database
nobel_coll_names = client.nobel.list_collection_names()
print(nobel_coll_names)

# Connect to the "nobel" database
db = client.nobel

# Retrieve sample prize and laureate documents
prize = db.prizes.find_one()
laureate = db.laureates.find_one()

# Print the sample prize and laureate documents
print(prize)
print(laureate)
print(type(laureate))

# Get the fields present in each type of document
prize_fields = list(prize.keys())
laureate_fields = list(laureate.keys())

print(prize_fields)
print(laureate_fields)

#Filters as subdocuments
filter_doc = {
    'born': '1845-03-27',
    'diedCountry': 'Germany',
    'gender': 'male',
    'surname': 'Rontgen'
}
db.laureates.count_documents(filter_doc)

#Simple filter documents
db.laureates.count_documents({'gender': 'female'})
db.laureates.count_documents({'diedCountry': 'France'})
db.laureates.count_documents({'bornCity': 'Warsaw'})

#Query operators
#value in a range
db.laureates.count_documents({
    'diedCountry': {
        '$in': ['France', 'USA']}})
#not equal
db.laureates.count_documents({
    'diedCountry': {
        '$ne': 'France'}})
#comparison (strings are compared lexicographically)
db.laureates.count_documents({
    'diedCountry': {
        '$gt': 'Belgium',
        '$lte': 'USA'}})

#Exercises:
# Create a filter for laureates who died in the USA
criteria = {'diedCountry': 'USA'}

# Save the count of these laureates
count = db.laureates.count_documents(criteria)
print(count)

# Create a filter for laureates who died in the USA but were born in Germany
criteria = {'diedCountry': 'USA', 
            'bornCountry': 'Germany'}

# Save the count
count = db.laureates.count_documents(criteria)
print(count)

# Create a filter for Germany-born laureates who died in the USA and with the first name "Albert"
criteria = {'diedCountry': 'USA', 
            'bornCountry': 'Germany', 
            'firstname': 'Albert'}

# Save the count
count = db.laureates.count_documents(criteria)
print(count)

# Save a filter for laureates born in the USA, Canada, or Mexico
criteria = { 'bornCountry': 
                { "$in": ['USA', 'Canada', 'Mexico']}
             }

# Count them and save the count
count = db.laureates.count_documents(criteria)
print(count)

# Save a filter for laureates who died in the USA and were not born there
criteria = { 'diedCountry': 'USA',
               'bornCountry': { "$ne": 'USA'}, 
             }

# Count them
count = db.laureates.count_documents(criteria)
print(count)

#. notation
#functional density
db.laureates.find_one({
    'firstname': 'Walter',
    'surname': 'Kohn'})

#prizes field is an array
db.laureates.count_documents({
    "prizes.affiliations.name": (
        "University of California")})
db.laureates.count_documents({
    "prizes.affiliations.city": (
        "Berkeley, CA")})

# if a field doesn't exist
db.laureates.find_one({'surname': 'Naipaul'})
#counts how many laureates don't have a bornCountry field
db.laureates.count_documents({'bornCountry': {'$exists': False}})

# to check if array elements are blank (counts those that are non-empty)
db.laureates.count_documents({'prizes.0': {'$exists': True}})

# Filter for laureates born in Austria with non-Austria prize affiliation
criteria = {'bornCountry': 'Austria', 
              'prizes.affiliations.country': {"$ne": 'Austria'}}

# Count the number of such laureates
count = db.laureates.count_documents(criteria)
print(count)

# Filter for documents without a "born" field
criteria = {'born': {'$exists': False}}

# Save count
count = db.laureates.count_documents(criteria)
print(count)

# Filter for laureates with at least three prizes
criteria = {"prizes.2": {'$exists': True}}

# Find one laureate with at least three prizes
doc = db.laureates.find_one(criteria)

# Print the document
print(doc)

#distinct() method
#collects set of values assigned to a field across all documents
db.laureates.distinct('gender')
#distance is efficient if there is an index on the field, otherwise may not be
#using . notation
db.laureates.find_one({'prizes.2': {'$exists': True}})
db.laureates.distinct('prizes.category')

#Exercises:
# Countries recorded as countries of death but not as countries of birth
countries = set(db.laureates.distinct('diedCountry')) - set(db.laureates.distinct('bornCountry'))
print(countries)

# The number of distinct countries of laureate affiliation for prizes
count = len(db.laureates.distinct('prizes.affiliations.country'))
print(count)

#prefiltering distinct values
db.laureates.find_one({'prizes.share': '4'})
#shows all categories
db.laureates.distinct('prizes.category')
#finds all with a 1/4 share of prize
list(db.laureates.find({'prizes.share': '4'}))
#shows categories with 1/4 share of prize
db.laureates.distinct('prizes.category', {'prizes.share': '4'})
#returns prize categories filterd for 1/4 share (same output)
db.prizes.distict('category', {'laureates.share': '4'})
#prize categories where people have won more than one prize
db.laureates.count_documents({'prizes.1': {'$exists': True}})
db.laureates.distinct('prizes.category', {'prizes.1': {'$exists': True}})

#Exercises:
db.laureates.distinct('prizes.affiliations.country', {'bornCountry': 'USA'})

# Save a filter for prize documents with three or more laureates
criteria = {'laureates.2': {'$exists': True}}

# Save the set of distinct prize categories in documents satisfying the criteria
triple_play_categories = set(db.prizes.distinct('category', criteria))

# Confirm literature as the only category not satisfying the criteria.
assert set(db.prizes.distinct('category')) - triple_play_categories == {'literature'}

#Matching array fields
db.laureates.count_documents({'prizes.category': 'physics'})
#matches on any member of the array
db.laureates.find({'nicknames': 'JB'})
#different than {'nicknames': ['JB']}

db.laureates.count_documents({'prizes.category': {'$ne': 'physics'}})
#returns larueates with at least one prize in these three categories
db.laureates.count_documents({'prizes.category': {'$in': ['physics', 'chemistry', 'medicine']}})
#unshared prizes in physics (this one doesn't work)
db.laureates.count_documents({'prizes': {'category': 'physics', 'share': '1'}})
#better, but still not right
db.laureates.count_documents({'prizes.category':'physics', 'prizes.share': '1'})
#best to use elemMatch - this returns unshared prizes in physics
db.laureates.count_documents({'prizes': {'$elemMatch': {'category': 'physics', 'share': '1'}}})
#same as above but from before 1945
db.laureates.count_documents({'prizes': {'$elemMatch': {'category': 'physics', 'share': '1', 'year': {'$lt': '1945'}}}})

#exercises:
#unshared prizes after 1945
db.laureates.count_documents({
    "prizes": {"$elemMatch": {
        "category": "physics",
        "share":  "1",
        "year": {"$gt": "1945"}}}})
#shared prizes after 1945
db.laureates.count_documents({
    "prizes": {"$elemMatch": {
        "category": "physics",
        "share": {"$ne": "1"},
        "year": {"$gt": "1945"}}}})

# Save a filter for laureates with unshared prizes
unshared = {
    "prizes": {"$elemMatch": {
        "category": {"$nin": ["physics", "chemistry", "medicine"]},
        "share": "1",
        "year": {"$gte": "1945"},
    }}}

# Save a filter for laureates with shared prizes
shared = {
    "prizes": {"$elemMatch": {
        "category": {"$nin": ["physics", "chemistry", "medicine"]},
        "share": {"$ne": "1"},
        "year": {"$gte": "1945"},
    }}}

ratio = db.laureates.count_documents(unshared) / db.laureates.count_documents(shared)
print(ratio)

# Save a filter for organization laureates with prizes won before 1945
before = {
    'gender': 'org',
    'prizes.year': {'$lt': "1945"},
    }

# Save a filter for organization laureates with prizes won in or after 1945
in_or_after = {
    'gender': 'org',
    'prizes.year': {'$gte': "1945"},
    }

n_before = db.laureates.count_documents(before)
n_in_or_after = db.laureates.count_documents(in_or_after)
ratio = n_in_or_after / (n_in_or_after + n_before)
print(ratio)

#Filtering with regex
db.laureates.distinct('bornCountry', {'bornCountry': {'$regex': 'Poland'}})
#flag options
case_sensitive = db.laureates.distinct('bornCountry', {'bornCountry': {'$regex': 'Poland'}})
#i option - case insentivie matching
case_insensitive = db.laureates.distinct('bornCountry', {'bornCountry': {'$regex': 'poland', '#options': 'i'}})
assert set(case_sensitive) == set(case_insensitive)
#this is a big import though, so may slow things down
from bson.regex import Regex
db.laureates.distinct('bornCountry', {'bornCountry': Regex('poland', 'i')})

import re
db.laureates.distinct('bornCountry', {'bornCountry': re.compile('poland', re.I)})

#beginning and ending and escaping
#to match beginning
db.laureates.distinct('bornCountry', {'bornCountry': Regex('^Poland')})
# to escape a character \
db.laureates.distinct('bornCountry', {'bornCountry': Regex('^Poland \(now')})
#to match end
db.laureates.distinct('bornCountry', {'bornCountry': Regex('now Poland\)$')})

#exercises:
db.laureates.count_documents({"firstname": Regex('^G'), "surname": Regex('^S')})

from bson.regex import Regex

# Filter for laureates with "Germany" in their "bornCountry" value
criteria = {"bornCountry": Regex('Germany')}
print(set(db.laureates.distinct("bornCountry", criteria)))

# Filter for laureates with a "bornCountry" value starting with "Germany"
criteria = {"bornCountry": Regex('^Germany')}
print(set(db.laureates.distinct("bornCountry", criteria)))

# Fill in a string value to be sandwiched between the strings "^Germany " and "now"
criteria = {"bornCountry": Regex("^Germany " + "\(" + "now")}
print(set(db.laureates.distinct("bornCountry", criteria)))

#Filter for currently-Germany countries of birth. Fill in a string value to be sandwiched between the strings "now" and "$"
criteria = {"bornCountry": Regex("now " + "Germany\)" + "$")}
print(set(db.laureates.distinct("bornCountry", criteria)))

# Save a filter for laureates with prize motivation values containing "transistor" as a substring
criteria = {'prizes.motivation': Regex('transistor')}

# Save the field names corresponding to a laureate's first name and last name
first, last = 'firstname', 'surname'
print([(laureate[first], laureate[last]) for laureate in db.laureates.find(criteria)])

#Projection
#reducing data to few dimensions
#includes only prizes.affiliations excludes _id
docs = db.laureates.find(filter={},
        projection={'prizes.affiliations': 1, '_id': 0})
#for each field we want to include in the projection, we give it a value of 1
#_id is included by default
#data type is Cursor
#you can conert to a list and slice
list(docs)[:3]
#what about missing fields?
#rather than return an error, mongo returns a document without those fields
docs = db.laureates.find(
    filter={'gender': 'org'},
    projection=['bornCountry', 'firstname'])
#only projected fields that exist are returned
#simple aggregation
docs = db.laureates.find({}, ['prizes'])
n_prizes = 0
for doc in docs:
    n_prizes += len(doc['prizes'])
print(n_prizes)

#if you use a projection than you only have to iterate through that field
#using a comprehension
sum([len(doc['prizes']) for doc in docs])

#exercises:
db.laureates.find(filter={'category': 'physics', 'year':'1903'}, projection=['firstname', 'surname'])

# Find laureates whose first name starts with "G" and last name starts with "S"
docs = db.laureates.find(
       filter= {"firstname" : {"$regex" : "^G"},
                  "surname" : {"$regex" : "^S"}  })
# Print the first document 
print(docs[0])

# Use projection to select only firstname and surname
docs = db.laureates.find(
        filter= {"firstname" : {"$regex" : "^G"},
                 "surname" : {"$regex" : "^S"}  },
	projection=['firstname', 'surname']  )

# Print the first document 
print(docs[0])

# Use projection to select only firstname and surname
docs = db.laureates.find(
       filter= {"firstname" : {"$regex" : "^G"},
                "surname" : {"$regex" : "^S"}  },
   projection= ["firstname", "surname"]  )

# Iterate over docs and concatenate first name and surname
full_names = [doc["firstname"] + " " + doc["surname"] for doc in docs]

# Print the full names
print(full_names)

# Save documents, projecting out laureates share
prizes = db.prizes.find({}, ['laureates.share'])

# Iterate over prizes
for prize in prizes:
# Initialize total share
    total_share = 0
# Iterate over laureates for the prize
for laureate in prize["laureates"]:
# add the share of the laureate to total_share
    total_share += 1 / float(laureate["share"])
# Print the total share 
    print(total_share) 

#Sorting
#can sort post-query with Python, which may be performant (if stored as local cache)
docs = list(db.prizes.find({"category": "physics"}, ["year"]))
print([doc["year"] for doc in docs][:5])
#to sort the documents
from operator import itemgetter
docs = sorted(docs, key=itemgetter("year"))
print([doc["year"] for doc in docs][:5])
#to sort in reverse (descending order)
docs = sorted(docs, key=itemgetter("year"), reverse=True)
print([doc["year"] for doc in docs][:5])
#sorting in query (ascending)
cursor = db.prizes.find({'category': 'physics'}, ['year'], sort=[('year', 1)])
print([doc["year"] for doc in cursor][:5])
#sorting in query (descending)
cursor = db.prizes.find({'category': 'physics'}, ['year'], sort=[('year', -1)])
print([doc["year"] for doc in cursor][:5])

#primary and secondary sorting
for doc in db.prizes.dinf:
    {'year': {'$gt': '1966', '$lt': '1970'}},
    ['catefory', 'year'],
    sort=[('year, 1'), ('category', -1)]
    print('{year} {category'.format(**doc))

#if you sort in MongoDB shell, you use Javascript
#JS objects retain key order as entered
#pymongo requires a list of tuples instead of a dictionary (it wants to retain order)
#Exercises:
docs = list(db.laureates.find(
{"born": {"$gte": "1900"}, "prizes.year": {"$gte": "1954"}},
{"born": 1, "prizes.year": 1, "_id": 0},
sort=[("prizes.year", 1), ("born", -1)]))
for doc in docs[:5]:
    print(doc)

from operator import itemgetter

def all_laureates(prize): 
# sort the laureates by surname
    sorted_laureates = sorted(prize['laureates'], key=itemgetter('surname'))
# extract surnames
    surnames = [laureate['surname'] for laureate in sorted_laureates]
# concatenate surnames separated with " and " 
    all_names = " and ".join(surnames)
    return all_names

# test the function on a sample doc
print(all_laureates(sample_prize))

# find physics prizes, project year and first and last name, and sort by year
docs = db.prizes.find(
filter= {'category': 'physics'}, 
projection= ['year', 'laureates.firstname', 'laureates.surname'], 
sort= [("year", 1)])

# print the year and laureate names (from all_laureates)
for doc in docs:
    print("{year}: {names}".format(year=doc['year'], names=all_laureates(doc)))

# original categories from 1901
original_categories = db.prizes.distinct('category', {'year': '1901'})
print(original_categories)

# project year and category, and sort
docs = db.prizes.find(
filter={},
projection = {'year': 1, 'category': 1, '_id': 0},
sort=[('year', -1),('category', 1)]
)

#print the documents
for doc in docs:
    print(doc)

#Indexes - speed up queries
#can index fields using the values of the fields
#when to use 
#queries with high specificity 
#large documents
#large collections
#Guaging performance before indexing
#can use %%timeit in Jupyter notebook 
#create index method
db.prizes.create_index([('year', 1)])
#direction (1 is ascending, -1 is descending)
docs = list(db.prizes.find({'year': "1901"}))
#compound index (multi-field)
db.prizes.create_index([('category', 1), ('year', 1)])
#index covering a query with projection (this can lead to big performance gains for queries used often)
list(db.prizes.find({'category': 'economics'}, {'year': 1, '_id': 0}))
#index covering a query with projection and sorting
db.prizes.find({'category': 'economics'},
{'year':1, '_id': 0},
sort=[('year', 1)])
#index information -helps confirm which indexes exist
db.laureates.index_information()
#explain method - will show indexes
db.laureates.find({'firstname': 'Marie'}, {'bornCountry': 1, '_id':0}).explain()

#exercies:
# Specify an index model for compound sorting
index_model = [("category", 1), ("year", -1)]
db.prizes.create_index(index_model)

# Collect the last single-laureate year for each category
report = ""
for category in sorted(db.prizes.distinct("category")):
    doc = db.prizes.find_one(
    {'category': category, "laureates.share": "1"},
    sort=[('year', -1)]
)
report += "{category}: {year}\n".format(**doc)

print(report)

from collections import Counter

# Ensure an index on country of birth
db.laureates.create_index([("bornCountry", 1)])

# Collect a count of laureates for each country of birth
n_born_and_affiliated = {
    country: db.laureates.count_documents({
    "bornCountry": country,
    "prizes.affiliations.country": country})}
for country in db.laureates.distinct("bornCountry"):
    five_most_common = Counter(n_born_and_affiliated).most_common(5)
    print(five_most_common)

#Limits
for doc in db.prizes.find({}, ['laureates.share']):
    share_is_three = [laureate['share']== "3"
        for laureate in doc['laureates']]
    assert all(share_is_three) or not any (share_is_three)

for doc in db.prizes.find({'laureates.share': "3"}):
    print('{year} {category}'.format(**doc))
#this limits the number of results
for dob in db.prizes.find({'laureates.share'}, limit=3):
    print('{year} {category}'.format(**doc))
#can also skip
for dob in db.prizes.find({'laureates.share'}, skip=3, limit=3):
    print('{year} {category}'.format(**doc))
#can chain methods to a cursor (sort, skip, limit)
for doc in db.prizes.find({'laureates.share': '3'}).limit(3):
    print('{year} {category}'.format(**doc))
#with both skip and limit
for doc in db.prizes.find({'laureates.share': '3'}).skip(3).limit(3):
    print('{year} {category}'.format(**doc))
#can alter the sorting on a cursor
for doc in db.prizes.find({'laureates.share': '3'}).sort([('year',1)]).skip(3).limit(3):
    print('{year} {category}'.format(**doc))
#simpler sort of sorts
cursor1 = (db.prizes.find({'laureates.share': '3'}).skip(3).limit(3).sort([('year', 1)]))
cursor2 = (db.prizes.find({'laureates.share': '3'}).skip(3).limit(3).sort('year', 1))
cursor3 = (db.prizes.find({'laureates.share': '3'}).skip(3).limit(3).sort('year'))
#all of these yield the same sequence of documents
#find_one will do something different
doc = db.prizes.find_one({'laureates.share': '3'}, skip=3, sort=[('year', 1)])
print('{year} {category}'.format(**doc))
#can't use cursor method with find_one
#if you use limit twice, the second one overrides the first

#exercises
from pprint import pprint

# Fetch prizes with quarter-share laureate(s)
filter_ = {'laureates.share': '4'}

# Save the list of field names
projection = ['category', 'year', 'laureates.motivation']

# Save a cursor to yield the first five prizes
cursor = db.prizes.find(filter_, projection).sort('year').limit(5)
pprint(list(cursor))

# Write a function to retrieve a page of data
def get_particle_laureates(page_number=1, page_size=3):
    if page_number < 1 or not isinstance(page_number, int):
        raise ValueError("Pages are natural numbers (starting from 1).")
    particle_laureates = list(
        db.laureates.find(
            {'prizes.motivation': {'$regex': "particle"}},
            ["firstname", "surname", "prizes"])
        .sort([('prizes.year', 1), ('surname', 1)])
        .skip(page_size * (page_number - 1))
        .limit(page_size))
    return particle_laureates

# Collect and save the first nine pages
pages = [get_particle_laureates(page_number=page) for page in range(1,9)]
pprint(pages[0])

#Aggregation stages
#queries have implicit stages
cursor = db.laureates.find(filter={'bornCountry': 'USA'},
            projection={'prizes.year': 1},
            limit=3)
for doc in cursor:
    print(doc['prizes'])
#aggregation pipeline is a list, a sequence of stages
#same output as aggregation
cursor = db.laureates.aggregate([
    {'$match': {'bornCountry': 'USA'}},
    {'$project': {'prizes.year': 1}},
    {'$limit': 3}
])
for doc in cursor:
    print(dob['prizes'])
#can sort and skip too
from collections import OrderedDict

cursor = db.laureates.aggregate([
    {'$match': {'bornCountry': 'USA'}},
    {'$project': {'prizes.year': 1, '_id': 0}},
    {'$sort': OrderedDict([('prizes.year', 1)])},
    {'$skip': 1},
    {'$limit': 3}
])
#can count
list(db.laureates.aggregate([
    {'$match': {'bornCountry': 'USA'}},
    {'$count': 'n_USA-born-laureastes'}
]))
#can do it this way too
db.laureates.count_documents({'bornCountry': 'USA'})

#Exercises:
# Translate cursor to aggregation pipeline
pipeline = [
    {'$match': {'gender': {'$ne': 'org'}}},
    {'$project': {'bornCountry': 1, 'prizes.affiliations.country': 1}},
    {'$limit': 3}
]

for doc in db.laureates.aggregate(pipeline):
    print("{bornCountry}: {prizes}".format(**doc))

from collections import OrderedDict
from itertools import groupby
from operator import itemgetter

original_categories = set(db.prizes.distinct("category", {"year": "1901"}))

# Save an pipeline to collect original-category prizes
pipeline = [
    {"$match": {"category": {"$in": list(original_categories)}}},
    {"$project": {"category": 1, "year": 1}},
    {"$sort": OrderedDict([("year", -1)])}
]
cursor = db.prizes.aggregate(pipeline)
for key, group in groupby(cursor, key=itemgetter("year")):
    missing = original_categories - {doc["category"] for doc in group}
    if missing:
        print("{year}: {missing}".format(year=key, missing=", ".join(sorted(missing))))

#aggregation operators and grouping
#Field paths
#expression object {field1: <expression1>, ...}
db.laureates.aggregate([{'$project': {'prizes.share': 1}}]).next()
#another way which will print differently
db.laureates.aggregate([{'$project': {'n_prizes': {'$size': '$prizes'}}}]).next()
#field path is prizes
#expression is {'$size': '$prizes'}
#operator expression is {'$size': '$prizes'}
#applies the operator to one or more arguments and returns a value
db.laureates.aggregate([{'$project': {'n_prizes': {'$size': ['$prizes']}}}]).next()
#returns the same result (can omit the brackets if you have only one parameter)
db.laureates.aggregate([{'$project': {'solo_winner': {'$in': ['1', 'prizes.share']}}}]).next()
#implementing .distinct()
list_1 = db.laureates.distinct('bornCountry')
list_2 = [doc['_id'] for doc in db.laureates.aggregate([
    {'$group': {'_id': '$bornCountry'}}
])]
set(list_2) - {None} == set(list_1)
#$group must map _id which must be unique
#how many prizes have been awarded in total
list(db.laureates.aggregate([
    {'$project': {'n_prizes': {'$size': '$priaxes'}}},
    {'$group': {'_id': None, 'n_prizes_total': {'$sum': '$n_prizes'}}}
]))
#this creates one document with the n_prizes_total expression
#$sum operator acts as an accumulator in $group stage
#this saves a lot of time and bandwidth for big datasets
#field paths use $ at the beginning to distinguish them from strings
#JSON sets are delimited by [] just like lists

#Exercises:
# Count prizes awarded (at least partly) to organizations as a sum over sizes of "prizes" arrays.
pipeline = [
    {'$match': {'gender': "org"}},
    {"$project": {"n_prizes": {"$size": '$prizes'}}},
    {"$group": {"_id": None, "n_prizes_total": {"$sum": 'n_prizes'}}}
]

print(list(db.laureates.aggregate(pipeline)))

from collections import OrderedDict

original_categories = sorted(set(db.prizes.distinct("category", {"year": "1901"})))
pipeline = [
    {"$match": {"category": {"$in": original_categories}}},
    {"$project": {"category": 1, "year": 1}},
    
    # Collect the set of category values for each prize year.
    {"$group": {"_id": "$year", "categories": {"$addToSet": "$category"}}},
    
    # Project categories *not* awarded (i.e., that are missing this year).
    {"$project": {"missing": {"$setDifference": [original_categories, "$categories"]}}},
    
    # Only include years with at least one missing category
    {"$match": {"missing.0": {"$exists": True}}},
    
    # Sort in reverse chronological order. Note that "_id" is a distinct year at this stage.
    {"$sort": OrderedDict([("_id", -1)])},
]
for doc in db.prizes.aggregate(pipeline):
    print("{year}: {missing}".format(year=doc["_id"],missing=", ".join(sorted(doc["missing"]))))

#Array fields with $unwind
#how to access array elements during aggregation
list(db.prizes.agggregate([
    {'$project': {'n_laureates': {'$size': '$laureates'},
        'year': 1, 'category': 1, '_id': 0}}
]))
#returns year, category and total for each (ie three physics prizes in 2018)
list(db.prizes.agggregate([
    {'$project': {'n_laureates': {'$size': '$laureates'},
        'category': 1}},
    {'$group': {'_id': '$category', 'n_laureates': {'$sum': '$n_laureates'}}},
    {'$sort': {'n_laureates': -1}}
]))
#returns total count of laureates by category (year is removed)
#How to $unwind
list(db.prizes.agggregate([
    {'$unwind': '$laureates'},
    {'$project': {'_id': 0, 'year': 1, 'category': 1,
        'laureates.surname': 1, 'laureates.share': 1}},
    {'$limit': 3}
]))
#outputs one document per array element
#to renormalize
list(db.prizes.agggregate([
    {'$unwind': '$laureates'},
    {'$project': {'year': 1, 'category': 1, 'laureates.id': 1}},
    {'$group': {'_id': {'$concat': ['$category', ':', '$year']},
        'laureate_ids': {'$addToSet': '$laureates.id'}}},
    {'$limit': 5}
]))

#this unwinds and counts documents
list(db.prizes.agggregate([
    {'$unwind': '$laureates'},
    {'$group': {'_id': '$category', 'n_laureates': {'$sum': 1}}},
    {'$sort': {'n_laureates': -1}},
]))
#$loopup usually accompanies $unwind
list(db.prizes.agggregate([
    {'$match': {'category': 'economics'}},
    {'$unwind': '$laureates'},
    {'$lookup': {'from': 'laureates', 'foreignField': 'id',
            'localField': 'laureates.id', 'as': 'laureate_bios'}},
    {'$unwind': '$laureate_bios'},
    {'$group': {'_id': None, 'bornCountries': {'$addToSet': '$laureate_bios.bornCountry'}}}
]))
#you can do this simpler:
bornCountries = db.laureates.distinct(
    'bornCountry', {'prizes.category': 'economics'})
assert set(bornCountries) == set(agg[0]['bornCountries'])

key_ac = "prizes.affiliations.country"
key_bc = "bornCountry"
pipeline = [
    {"$project": {key_bc: 1, key_ac: 1}},

    # Ensure a single prize affiliation country per pipeline document
    {'$unwind': "$prizes"},
    {'$unwind': "$prizes.affiliations"},

    # Ensure values in the list of distinct values (so not empty)
    {"$match": {key_ac: {'$in': db.laureates.distinct(key_ac)}}},
    {"$project": {"affilCountrySameAsBorn": {
        "$gte": [{"$indexOfBytes": ["$"+key_ac, "$"+key_bc]}, 0]}}},

    # Count by "$affilCountrySameAsBorn" value (True or False)
    {"$group": {"_id": "$affilCountrySameAsBorn",
                "count": {"$sum": 1}}},
]
for doc in db.laureates.aggregate(pipeline): print(doc)

pipeline = [
    # Unwind the laureates array
    {"$unwind": "$laureates"},
    {"$lookup": {
        "from": "laureates", "foreignField": "id",
        "localField": "laureates.id", "as": "laureate_bios"}},

    # Unwind the new laureate_bios array
    {"$unwind": "$laureate_bios"},
    {"$project": {"category": 1,
                  "bornCountry": "$laureate_bios.bornCountry"}},

    # Collect bornCountry values associated with each prize category
    {"$group": {"_id": "$category",
                "bornCountries": {"$addToSet": "$bornCountry"}}},

    # Project out the size of each category's (set of) bornCountries
    {"$project": {"category": 1,
                  "nBornCountries": {"$size": "$bornCountries"}}},
    {"$sort": {"nBornCountries": -1}},
]
for doc in db.prizes.aggregate(pipeline): print(doc)

#addFields to aid analysis
docs = list(db.laureates.aggregate([ 
    {'$match': {'died': {'$gt': '1700'}, 'born': {'$gt': '1700'}}},
    {'$project': {'died': {'$dateFromString': {'dateString': '$died'}},
            'born': {'$dateFromString': {'dateString': '$born'}}}}
]))
#some dob only have a year, so we get an error
docs = list(db.laureates.aggregate([ 
    {'$match': {'died': {'$gt': '1700'}, 'born': {'$gt': '1700'}}},
    {'$addFields': {'bornArray': {'$split': ['$born', '-']},
                    'diedArray': {'$split': ['$died', '-']}}},
    {'$addFields': {'born': {'$cond': [
        {'$in': ['00', '$bornArray']},
        {'$concat': [{'$arrayElemAt': ['$bornArray', 0]}, '-01-01']},
    ]}}},
    {'$project': {'died': {'$dateFromString': {'dateString': '$died'}},
            'born': {'$dateFromString': {'dateString': '$born'}}}},
    {'$project': {'years': {'$floor': {'$divide': [
        {'$subtract': ['$died', '$born']},
        31557600000
    ]}}}},
    {'$bucket': {'groupBy': '$years',
            'boundaries': list(range(30,120,10))}}
]))
for doc in docs: print(doc)

#Exercises:
pipeline = [
    # Limit results to people; project needed fields; unwind prizes
    {"$match": {"gender": {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
  
    # Count prizes with no country-of-birth affiliation
    {"$addFields": {"bornCountryInAffiliations": {"$in": ["$bornCountry", "$prizes.affiliations.country"]}}},
    {"$match": {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]

print(list(db.laureates.aggregate(pipeline)))

pipeline = [
    {"$match": {"gender": {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
    {"$addFields": {"bornCountryInAffiliations": {"$in": ["$bornCountry", "$prizes.affiliations.country"]}}},
    {"$match": {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]

# Construct the additional filter stage
added_stage = {"$match": {"prizes.affiliations.country": {"$in": db.laureates.distinct("prizes.affiliations.country")}}}

# Insert this stage into the pipeline
pipeline.insert(3, added_stage)
print(list(db.laureates.aggregate(pipeline)))

