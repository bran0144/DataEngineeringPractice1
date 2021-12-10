# Data Projects need: storage, compute resources, alerts
# AWS services are granular (they work together or on their own)
# To interact with aws in Python, we use Boto3 library
import boto3
import pandas as pd

s3 = boto3.client('s3',         #service name
            region_name='us-east-1',
            aws_access_key_id=AWS_KEY_ID,
            aws_secrete_access_key=AWS_SECRET)

response = s3.list_buckets()
# service name can be any of the 100+ available AWS services
# after signing up for AWS (creates a root user), you can create IAM users
# IAM - Identity access management service
# create IAM subusers to control access to AWS resources in the account
# Credentials (the key/secret combo and are what authenticate IAM users)
# S3 - simple storage service (stores files in cloud)
# SNS - simple notification service (sends emails and texts to alert subscribers
#   based on events and conditions in data pipelines
# Comprehend - performs sentiment analysis on blocks of text
# Rekogntion - extracts text from images and looks for cats in a picture

# Buckets
# S3 - main components: buckets, objects
# buckets are like directories
# objects are like files
# Buckets:
#   have their own permission policies
#   can be configured to act as folders for a static website
#   can generate logs and write them to a different bucket
#   contain objects
#   names have to be unique across all of s3
# Objects:
#   can be anything (image, video, csv, log)
#   must be stored in a bucket
# Boto3 lets you:
#   create a bucket
#   list buckets
#   delete buckets

s3 = boto3.client('s3',         #service name
            region_name='us-east-1',
            aws_access_key_id=AWS_KEY_ID,
            aws_secrete_access_key=AWS_SECRET)
bucket = s3.create_bucket(Bucket='gid-requests')
bucket_response = s3.list_buckets() #this will include additional metatdata
buckets = bucket_response['Buckets'] #this will be a dictionary of the bucket names
print(buckets)
response = s3.delete_bucket('gid-requests')

# Exercises:

# Create boto3 client to S3
s3 = boto3.client('s3', region_name='us-east-1', 
                         aws_access_key_id=AWS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET)

# Create the buckets
response_staging = s3.create_bucket(Bucket='gim-staging')
response_processed = s3.create_bucket(Bucket='gim-processed')
response_test = s3.create_bucket(Bucket='gim-test')

# Print out the response
print(response_staging)

# Get the list_buckets response
response = s3.list_buckets()

# Iterate over Buckets from .list_buckets() response
for bucket in response['Buckets']:
  
  	# Print the Name for each bucket
    print(bucket['Name'])

    # Delete the gim-test bucket
s3.delete_bucket(Bucket='gim-test')

# Get the list_buckets response
response = s3.list_buckets()

# Print each Buckets Name
for bucket in response['Buckets']:
    print(bucket['Name'])

    # Get the list_buckets response
response = s3.list_buckets()

# Delete all the buckets with 'gim', create replacements.
for bucket in response['Buckets']:
  if 'gim' in bucket['Name']:
      s3.delete_bucket(Bucket=bucket['Name'])
    
s3.create_bucket(Bucket='gid-staging')
s3.create_bucket(Bucket='gid-processed')
  
# Print bucket listing after deletion
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(bucket['Name'])

# Uploading and retrieving files
# Buckets have names 
    # (which is just a string)
    # unicue name is all of s3
    # can contain many objects
# Object have keys (full path from bucket root)
    # Unique to the bucket   
    # can only be in one parent bucket

s3.upload_file(
    Filename='gid_requests_2019_01_01.csv',
    Bucket='gid-requests',
    Key='gid_requests_2019_01-01.csv')

# Listing objects in a bucket
response = s3.list_objects(
    Bucket='gid-requests',
    MaxKeys=2,
    Prefix='gid_requests_2019_'
)
# Returns a Contents dictionary of objects and their info
# Includes key, lastmodified date, size, owner, id
# To return a single object:
response = s3.head_object(
    Bucket='gid-requests',
    Key='gid_requests_2018_12_30.csv')
# Does not return a Contents dict - just a dict of metadata 
# To download a fie
s3.download_file(
    Filename='gid_requests_downed.csv',
    Bucket='gid-requests',
    Key='gid_requests_2018_12_30.csv')
# To delete an object:
s3.delete_object(
    Bucket='gid-requests',
    Key='gid_requests_2018_12_30.csv')

# Exercises:
# Upload final_report.csv to gid-staging
s3.upload_file(Bucket='gid-staging',
              # Set filename and key
               Filename='final_report.csv', 
               Key='2019/final_report_01_01.csv')

# Get object metadata and print it
response = s3.head_object(Bucket='gid-staging', 
                       Key='2019/final_report_01_01.csv')

# Print the size of the uploaded object
print(response['ContentLength'])

# List only objects that start with '2018/final_'
response = s3.list_objects(Bucket='gid-staging', 
                           Prefix='2018/final_')

# Iterate over the objects
if 'Contents' in response:
  for obj in response['Contents']:
      # Delete the object
      s3.delete_object(Bucket='gid-staging', Key=obj['Key'])

# Print the remaining objects in the bucket
response = s3.list_objects(Bucket='gid-staging')

for obj in response['Contents']:
  	print(obj['Key'])

# Setting up permissions
# s3 won't let you download an s3 file with pandas
# need to first initialize the s3 client with credentials, then you can
s3.download_file(Filename='potholes.csv', Bucket='gid-requests', Key='potholes.csv')
# IAM - contols users access to AWS services, buckets, objects - good for multiuser
# Bucket policies - give control on the bucket and its objects - good for multiuser
# ACL - access control lists - set permissions on specific objects within a bucket
# Presigned URL - gives temporary access to an object
# ACL - entities attached to objects in S3
# ACL='public-read'
# ACL='private'
# when you upload a file, the default is private
s3.upload_file(
    Filename='potholes.csv', Bucket='gid-requests', Key='potholes.csv')
s3.put_object_acl(
    Bucket='gid-requests', Key='potholes.csv', ACL='public-read')

# Can set ACL on upload in a dictionary:
s3.upload_file(
    Filename='potholes.csv', 
    Bucket='gid-requests', 
    Key='potholes.csv', 
    ExtraArgs={'ACL': 'public-read'})
# If public, anyone can access it through URL
# https://{bucket}.s3.amazonaws.com/{key}
# Python can create the public URL
url = "https://{}.s3.amazonaws.com/{}".format(
    "gid-requests",
    "2019/potholes.csv"
)
# this can then be passed to pandas to read_csv
# Exercises:
# Upload the final_report.csv to gid-staging bucket
s3.upload_file(
  # Complete the filename
  Filename='./final_report.csv', 
  # Set the key and bucket
  Key='2019/final_report_2019_02_20.csv', 
  Bucket='gid-staging',
  # During upload, set ACL to public-read
  ExtraArgs = {
    'ACL': 'public-read'})

# List only objects that start with '2019/final_'
response = s3.list_objects(
    Bucket='gid-staging', Prefix='2019/final_')

# Iterate over the objects
for obj in response['Contents']:

    # Give each object ACL of public-read
    s3.put_object_acl(Bucket='gid-staging', 
                      Key=obj['Key'], 
                      ACL='public-read')
    
    # Print the Public Object URL for each object
    print("https://{}.s3.amazonaws.com/{}".format( 'gid-staging', obj['Key']))

# How to share private files? Need to balance access, security, and sharing
# One way: use s3.download_file and then read the csv from disc through pd.read_csv
# Does not work well if files changes often
# Can use .get_object()
obj = s3.get_object(Bucket='gid-requests', Key='2019/potholes.csv')
# returns similar metadata to the head_object method but also includes a StreamingBody
# this response does not download the whole object immediately
# pandas knows hwo to read StreamingBody
pd.read_csv(obj['Body'])

# can also grant access to s3 private objects temporarily using presigned_urls
# expire after a certain time period
# after uploading a file (default private):
share_url=s3.generate_presigned_url(
    ClientMethod="get_object",
    ExpiresIn=3600,             #grants access for 1 hour (3600 seconds)
    Params={'Bucket': 'gid-requests', 'Key': 'potholes.csv'}
)
pd.read_csv(share_url)
# Load multiple files into one DF
df_list = []
response = s3.list_objects(
    Bucket='gid-requests',
    Prefix='2019/')
request_files = response['Contents']

for file in request_files:
    obj = s3.get_object(Bucket='gid-requests', Key=file['Key'])
    obj_df = pd.read_csv(obj['Body'])
    df_list.append(obj_df)

df = pd.concat(df_list)

# Exercises:
# Generate presigned_url for the uploaded object
share_url = s3.generate_presigned_url(
  # Specify allowable operations
  ClientMethod='get_object',
  # Set the expiration time
  ExpiresIn=3600,
  # Set bucket and shareable object's name
  Params={'Bucket': 'gid-staging','Key': 'final_report.csv'}
)

# Print out the presigned URL
print(share_url)

df_list =  [ ] 

for file in response['Contents']:
    # For each file in response load the object from S3
    obj = s3.get_object(Bucket='gid-requests', Key=file['Key'])
    # Load the object's StreamingBody with pandas
    obj_df = pd.read_csv(obj['Body'])
    # Append the resulting DataFrame to list
    df_list.append(obj_df)

# Concat all the DataFrames with pandas
df = pd.concat(df_list)

# Preview the resulting DataFrame
df.head()

# Sharing files through a website
# S3 is able to share as html pages
# can use the .to_html pandas method
df.to_html('table_agg.html', 
    columns['service_name', 'request_count', 'info_link'],
    render_links=True,
    border=0)
# render_links makes URLs clickable
# columns parameter lets you pass only the columns you want rendered
# border =0 (will remove the border, 1 will show it)
s3.upload_file(
    Filename='./table_agg.html',
    Bucket='datacamp-website',
    Key='table.html',
    ExtraArgs={
        'ContentType': 'text/html',
        'ACL': 'public-read'}
)
# To upload images (like a Python chart):
s3.upload_file(
    Filename='./plot_image.png',
    Bucket='datacamp-website',
    Key='plot_image.png',
    ExtraArgs={
        'ContentType': 'image/png',
        'ACL': 'public-read'}
)
# IANA - lists the file types based on their extensions
# Can generate an index page
r = s3.list_objects(Bucket='gid-reports', Prefix='2019/')
objects_df = pd.DataFrame(r['Contents'])
base_url="http//datacamp-website.d2.amazonaws.com/"
objects_df['Link'] = base_url + objects_df['Key']
objects_df.to_html('report_listing.html',
    columns=['Link', 'LastModified', 'Size'],
    render_links=True)

s3.upload_file(
    Filename='./report_listing.html',
    Bucket='datacamp-website',
    Key='index.html',
    ExtraArgs={
        'ContentType': 'text/html',
        'ACL': 'public-read'}
)

# Exercises:
# Generate an HTML table with no border and selected columns
services_df.to_html('./services_no_border.html',
           # Keep specific columns only
           columns=['service_name', 'link'],
           # Set border
           border=0)

# Generate an html table with border and all columns.
services_df.to_html('./services_border_all_columns.html', 
           border=1)

# Upload the lines.html file to S3
s3.upload_file(Filename='lines.html', 
               # Set the bucket name
               Bucket='datacamp-public', Key='index.html',
               # Configure uploaded file
               ExtraArgs = {
                 # Set proper content type
                 'ContentType':'text/html',
                 # Set proper ACL
                 'ACL': 'public-read'})

# Print the S3 Public Object URL for the new file.
print("http://{}.s3.amazonaws.com/{}".format('datacamp-public', 'index.html'))

# Case Study
# Need to puck up raw data from the gid-requests bucket
df_list = []
response = s3.list_objects(
    Bucket='gid-requests',
    Prefix='2019_jan'
)
request_files = response['Contents']

for file in request_files:
    obj = s3.get_object(Bucket='gid-requests', Key=file['Key'])
    obj_df = pd.read_csv(obj['Body'])
    df_list.append(obj_df)          #should have a list of 31 DF's (one for each day of Jan)

df = pd.concat(df_list)

s3.upload_file(Filename='./jan_final_report.csv',
                Key='2019/jan/final_report.csv',
                Bucket='gid-reports',
                ExtraArgs= {'ACL': 'public-read'})
s3.upload_file(Filename='./jan_final_report.html',
                Key='2019/jan/final_report.html',
                Bucket='gid-reports',
                ExtraArgs= {'ContentType': 'text/html',
                'ACL': 'public-read'})
s3.upload_file(Filename='./jan_final_chart.html',
                Key='2019/jan/final_chart.html',
                Bucket='gid-reports',
                ExtraArgs= {'ContentType': 'text/html',
                'ACL': 'public-read'})

r = s3.list_objects(Bucket='gid-reports', Prefix='2019/')
objects_df = pd.DataFrame(r['Contents'])
base_url = "http://gid-reports.s3.amazonaws.com/"
objects_df['Link'] = base_url + objects_df['Key']

objects_df.to_html('report_listing.html',
            columns=['Link', 'LastModified', 'Size'],
            render_links=True)
s3.upload_file(Filename='./report_listing.html',
                Key='index.html',
                Bucket='gid-reports',
                ExtraArgs={'ContentType': 'text/html',
                'ACL': 'public-read'})

# Exercises:
df_list = [] 

# Load each object from s3
for file in request_files:
    s3_day_reqs = s3.get_object(Bucket='gid-requests', 
                                Key=file['Key'])
    # Read the DataFrame into pandas, append it to the list
    day_reqs = pd.read_csv(s3_day_reqs['Body'])
    df_list.append(day_reqs)

# Concatenate all the DataFrames in the list
all_reqs = pd.concat(df_list)

# Preview the DataFrame
all_reqs.head()

# Write agg_df to a CSV and HTML file with no border
agg_df.to_csv('./feb_final_report.csv')
agg_df.to_html('./feb_final_report.html', border=0)

# Upload the generated CSV to the gid-reports bucket
s3.upload_file(Filename='./feb_final_report.csv', 
	Key='2019/feb/final_report.html', Bucket='gid-reports',
    ExtraArgs = {'ACL': 'public-read'})

# Upload the generated HTML to the gid-reports bucket
s3.upload_file(Filename='./feb_final_report.html', 
	Key='2019/feb/final_report.html', Bucket='gid-reports',
    ExtraArgs = {'ContentType': 'text/html', 
                 'ACL': 'public-read'})

# List the gid-reports bucket objects starting with 2019/
objects_list = s3.list_objects(Bucket='gid-reports', Prefix='2019/')

# Convert the response contents to DataFrame
objects_df = pd.DataFrame(objects_list['Contents'])

# Create a column "Link" that contains Public Object URL
base_url = "http://gid-reports.s3.amazonaws.com/"
objects_df['Link'] = base_url + objects_df['Key']

# Preview the resulting DataFrame
objects_df.head()

# Write objects_df to an HTML file
objects_df.to_html('report_listing.html',
    # Set clickable links
    render_links=True,
	# Isolate the columns
    columns=['Link', 'LastModified', 'Size'])

# Overwrite index.html key by uploading the new file
s3.upload_file(
  Filename='./report_listing.html', Key='index.html', 
  Bucket='gid-reports',
  ExtraArgs = {
    'ContentType': 'text/html', 
    'ACL': 'public-read'
  })