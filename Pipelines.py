# Components of a data platform
# Data Lake - typically comprises several systems and is typically organized in several zones
# data from operational systems ends up in a "landing zone" (raw data)
# ingestion - getting data into the data lake
# clean zone - data from landing zone and cleaned (to prevent many similar transformations of data)
# business zone - applies ml algorithms to cleaned data (domain specific data)
# to move data from one zone to another and transform it, data pipelines are built
# pipelines can be triggered by external events like files be stored, or scheduled, or even manually
# ETL - extract, transform, load pipelines
# To navigate a data lake, a data catalog is typically provided

# Data ingestion with Singer
# writes scripts that move data
# Singer is a specification that uses JSON as the data exchange format
# taps - extraction scripts
# targets - loads scripts
# language independent
# can be mixed and matched to create smaller data pipelines
# Taps and targets communicate over streams:
    # schema(metadata)
    # state(process metadata)
    # record(data)
# Stream - a named virtual location to which you send messages that can be picked up at a downstream location
# different streams cna be used to partition data based on the topic
# Singer spec - first you describe the data by specifying its schema
# schema should be valid JSON
json_schema = {
  "properties": {"age": {"maximum": 130,
                          "minimum": 1,
                          "type": "integer"},
              "has_children": {"type": "boolean"},
              "id": {"type": "integer"},
              "name": {"type": "string"},
  "$id": "http://yourdomain.com/schemas/my_user_schema.json",
  "$scheme": "http//json-schema.org/draft-08/sceham#"}}

# id and schema are optional, but high recommended
import singer
singer.write_schema(schema=json_schema, 
                    stream_name='DC-employees',
                    key_properties=["id"])

# if there is no primary key, specify an empty list
# write_schema - wraps actual JSON scehma into a new JSON message and adds a few attributes
# json.dumps - transforms object into a string
# json.dump - writes the string to a file

# Import json
import json

database_address = {
  "host": "10.0.0.5",
  "port": 8456
}

# Open the configuration file in writable mode
with open("database_config.json", "w") as fh:
  # Serialize the object in this file handle
  json.dump(obj=database_address, fp=fh)

# Complete the JSON schema
schema = {'properties': {
    'brand': {'type': 'string'},
    'model': {'type': 'string'},
    'price': {'type': 'number'},
    'currency': {'type': 'string'},
    'quantity': {'type': 'integer', 'minimum': 1},  
    'date': {'type': 'string', 'format': 'date'},
    'countrycode': {'type': 'string', 'pattern': "^[A-Z]{2}$"}, 
    'store_name': {'type': 'string'}}}

# Write the schema
columns = ("id", "name", "age", "has_children")
users = {(1, "John", 20, False),
        (2, "Mary", 35, True),
        (3, "Sophia", 25, False)}
singer.write_schema(stream_name="products", schema=schema, key_properties=[])
singer.write_record(stream_name="DC-employees", 
               record=dict(zip(columns, users.pop())))


# Running an ingestion pipeline
# to convert a user into a Singer RECORD message, we use the "write_record" function

singer.write_record(stream_name="DC_employees",
            record=dict(zip(columns, users.pop())))
# stream_name needs to match the stream you specificed in a schema message
# Singer does a few more transformations that make better JSON than using **

# can use the unpacking operator (**) - which unpacks a dictionary into another one
fixed_dict={"type": "RECORD", "stream": "DC_employees"}
record_msg={**fixed_dict, "record": dict(zip(columns, users.pop()))}
print(json.dumps(record_msg))

# If you have a Singer target, that can parse the messages (along with write_schema
# and write_record), then you have a full ingestion pipeline (using | )
# write_records - can take more than one record

# python my_tap.py | target-csv
# this creates csv files from the json lines
# will put csv in the same directory from where you run the command
# python my_tap.py | target-csv --config userconfig.cfg
# usually they are package in python so your call would look like this:
# my-packaged-tap | target-csv --config userconfig.cfg

# Allows for modualr ingestion pipelines
# my-packaged-tap | target-google-sheets
# my-packaged-tap | target-postgresql --config conf.json
# just need to change taps and targets and you can ingest easily

# State messages
# good for when you only want to read newest values (you can emit state at the end of a 
#   successful run)
# then reuse that same message to only get the new messages
singer.write_state(value={"max-last-updated-on": some_variable})

# Exercises:
endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Create the web API’s URL
authenticated_endpoint = "{}/{}".format(endpoint, api_key)

# Get the web API’s reply to the endpoint
api_response = requests.get(authenticated_endpoint).json()
pprint.pprint(api_response)

# Create the API’s endpoint for the shops
shops_endpoint = "{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "shops")
shops = requests.get(shops_endpoint).json()
print(shops)

# Create the API’s endpoint for items of the shop starting with a "D"
items_of_specific_shop_URL = "{}/{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "items", "DM")
products_of_shop = requests.get(items_of_specific_shop_URL).json()
pprint.pprint(products_of_shop)

# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

for shop in requests.get(SHOPS_URL).json()["shops"]:
    # Write all of the records that you retrieve from the API
    singer.write_records(
      stream_name="products", # Use the same stream name that you used in the schema
      records=({**item, "store_name": shop}
               for item in retrieve_products(shop))
    )
