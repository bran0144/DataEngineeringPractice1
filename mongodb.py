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

