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
singer.write_schema(stream_name="products", schema=schema, key_properties=[])

singer.write_record(stream_name="DC-employees", 
               record=dict(zip(columns, users.pop())))