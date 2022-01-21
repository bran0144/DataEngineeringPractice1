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
