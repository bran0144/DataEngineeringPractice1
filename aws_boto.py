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

# SNS Topics
# Sending alerts is very important in data engineering
# Has publishers and subscribers
# Topics - every topic as an ARN (amazon resource name)
# Creating an SNS Topic:
sns = boto3.client(
    'sns',
    region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access=AWS_SECRET)

response = sns.create_topic(Name='city_alerts')
# returns an API reponse from AWS
topic_arn = response['TopicArn']
# or use this shortcut:
sns.create_topic(Name='city_alerts')['TopicArn']
# creating topics is idempotent - if we try to create a topic with the same name, it will return
# the topic that already exists (not an error)

reponse = sns.list_topics()
# returns a topics' key with a list of topics our user has access to

# Exercises:
# Initialize boto3 client for SNS
sns = boto3.client('sns', 
                   region_name='us-east-1', 
                   aws_access_key_id=AWS_KEY_ID, 
                   aws_secret_access_key=AWS_SECRET)

# Create the city_alerts topic
response = sns.create_topic(Name="city_alerts")
c_alerts_arn = response['TopicArn']

# Re-create the city_alerts topic using a oneliner
c_alerts_arn_1 = sns.create_topic(Name='city_alerts')['TopicArn']

# Compare the two to make sure they match
print(c_alerts_arn == c_alerts_arn_1)

# Create list of departments
departments = ['trash', 'streets', 'water']

for dept in departments:
  	# For every department, create a general topic
    sns.create_topic(Name="{}_general".format(dept))
    
    # For every department, create a critical topic
    sns.create_topic(Name="{}_critical".format(dept))

# Print all the topics in SNS
response = sns.list_topics()
print(response['Topics'])

# Get the current list of topics
topics = sns.list_topics()['Topics']

for topic in topics:
  # For each topic, if it is not marked critical, delete it
  if "critical" not in topic['TopicArn']:
    sns.delete_topic(TopicArn=topic['TopicArn'])
    
# Print the list of remaining critical topics
print(sns.list_topics()['Topics'])

# Subscription List (on GUI)
# each has a unique ID, protocol, and status
# There are several options for protocols (email, SMS, etc)
# endpoint - where message should be sent
# phone numbers are automatically confirmed
# for email, user has to click link to authorize the subscription
# To create an SMS subscription:
sns = boto3.client('sns',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_SECRET)

response = sns.subscribe(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts',
    Protocol = "SMS",
    Endpoint = '+13125551234')
# response is a dictionary that contains "SubscriptionArn" key
# To make an email subscription:
response = sns.subscribe(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts',
    Protocol = "email",
    Endpoint = 'data@datacamp.com')
# do not get a subcriptionArn key right away, you get a pending confirmation message
# to list subscriptions:
sns.list_subscriptions_by_topic(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts'
)
# returns a subscriptions key with a list of subscription dictionaries
# will include ARN if possible
# to list all subscriptions:
sns.list_subscriptions()['Subscriptions']
# deleting subscriptions:
sns.unsubscribe(SubscribeArn='arn:aws:sns:us-east-1:...')
# to delete all subscriptions that use SMS protocol
reponse = sns.list_subscriptions_by_topic(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts')
subs = response['Subscriptions']
for sub in subs:
    if sub['Protocol'] == 'sms':
        sns.unsubscribe(sub['SubscriptionArn'])

# Exercises:
# Subscribe Elena's phone number to streets_critical topic
resp_sms = sns.subscribe(
  TopicArn = str_critical_arn, 
  Protocol='sms', Endpoint="+16196777733")

# Print the SubscriptionArn
print(resp_sms['SubscriptionArn'])

# Subscribe Elena's email to streets_critical topic.
resp_email = sns.subscribe(
  TopicArn = str_critical_arn, 
  Protocol='email', Endpoint="eblock@sandiegocity.gov")

# Print the SubscriptionArn
print(resp_email['SubscriptionArn'])

# For each email in contacts, create subscription to street_critical
for email in contacts['Email']:
  sns.subscribe(TopicArn = str_critical_arn,
                # Set channel and recipient
                Protocol = 'email',
                Endpoint = email)

# List subscriptions for streets_critical topic, convert to DataFrame
response = sns.list_subscriptions_by_topic(
  TopicArn = str_critical_arn)
subs = pd.DataFrame(response['Subscriptions'])

# Preview the DataFrame
subs.head()

# List subscriptions for streets_critical topic.
response = sns.list_subscriptions_by_topic(
  TopicArn = str_critical_arn)

# For each subscription, if the protocol is SMS, unsubscribe
for sub in response['Subscriptions']:
  if sub['Protocol'] == 'sms':
	  sns.unsubscribe(SubscriptionArn=sub['SubscriptionArn'])

# List subscriptions for streets_critical topic in one line
subs = sns.list_subscriptions_by_topic(
  TopicArn=str_critical_arn)['Subscriptions']

# Print the subscriptions
print(subs)

# Pulishing to a topic 
# all subscribers will receive it
response = sns.publish(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts',
    Message = 'Body text of SMS or email',
    Subject = 'Subject line for email'      #not visible for SMS
)
# Can use stringf formatting to replace with variables from data
num_of_reports = 137
response = sns.publish(
    TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts',
    Message = 'There are {} reports outstanding'.format(num_of_reports),
    Subject = 'Subject line for email'      
)
# Can send a single SMS message, (don't need a topic or subscribers) (but can't send email this way)
response = sns.publish(
    PhoneNumber ='+131212345678',
    Message = 'Body text'
)

# Exercises:
# If there are over 100 potholes, create a message
if streets_v_count > 100:
  # The message should contain the number of potholes.
  message = "There are {} potholes!".format(streets_v_count)
  # The email subject should also contain number of potholes
  subject = "Latest pothole count is {}".format(streets_v_count)

  # Publish the email to the streets_critical topic
  sns.publish(
    TopicArn = str_critical_arn,
    # Set subject and message
    Message = message,
    Subject = subject
  )

# Loop through every row in contacts
for idx, row in contacts.iterrows():
    
    # Publish an ad-hoc sms to the user's phone number
    response = sns.publish(
        # Set the phone number
        PhoneNumber = str(row['Phone']),
        # The message should include the user's name
        Message = 'Hello {}'.format(row['Name'])
    )
   
    print(response)

# Case study
sns = boto3.client('sns',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_SECRET)

trash_arn = sns.create_topic(Name='trash_notifications')['TopicArn']
streets_arn = sns.create_topic(Name='streets_notifications')['TopicArn']

contacts = pd.read_csv('http://gid-staging.s3.amazonaws.com/contacts.csv')

def subscribe_user(user_row):
    if user_row['Department'] == 'trash':
        sns.subscribe(TopicArn=trash_arn, Protocol='sms', Endpoint=str(user_row['Phone']))
        sns.subscribe(TopicArn=trash_arn, Protocol='email', Endpoint=user_row['Email'])
    else:
        sns.subscribe(TopicArn=streets_arn, Protocol='sms', Endpoint=str(user_row['Phone']))
        sns.subscribe(TopicArn=streets_arn, Protocol='email', Endpoint=user_row['Email'])

contacts.apply(subscribe_user, axis=1)

# can do it this way, because it is public
df = pd.read_csv('http://gid-reports.s3.amazonaws.com/2019/feb/final_report.csv')
df.set_index('service_name', inplace=True)
trash_violations_count = df.at['Illegal Dumping', 'count']
street_violations_count = df.at['Pothole', 'count']

if trash_violations_count > 100:
    message = "Trash violations count is now {}".format(trash_violations_count)
    sns.publish(TopicArn = trash_arn,
                Message = message,
                Subject = "Trash Alert")

if street_violations_count > 30:
    message = "Street violations count is now {}".format(street_violations_count)
    sns.publish(TopicArn = streets_arn,
                Message = message,
                Subject = "Streets Alert")          

# Exercises:
dept_arns = {}

for dept in departments:
  # For each deparment, create a critical Topic
  critical = sns.create_topic(Name="{}_critical".format(dept))
  # For each department, create an extreme Topic
  extreme = sns.create_topic(Name="{}_extreme".format(dept))
  # Place the created TopicARNs into a dictionary 
  dept_arns['{}_critical'.format(dept)] = critical['TopicArn']
  dept_arns['{}_extreme'.format(dept)] = extreme['TopicArn']

# Print the filled dictionary
print(dept_arns)

for index, user_row in contacts.iterrows():
  # Get topic names for the users's dept
  critical_tname = '{}_critical'.format(user_row['Department'])
  extreme_tname = '{}_extreme'.format(user_row['Department'])
  
  # Get or create the TopicArns for a user's department.
  critical_arn = sns.create_topic(Name=critical_tname)['TopicArn']
  extreme_arn = sns.create_topic(Name=extreme_tname)['TopicArn']
  
  # Subscribe each users email to the critical Topic
  sns.subscribe(TopicArn = critical_arn, 
                Protocol='email', Endpoint=user_row['Email'])
  # Subscribe each users phone number for the extreme Topic
  sns.subscribe(TopicArn = extreme_arn, 
                Protocol='sms', Endpoint=str(user_row['Phone']))

if vcounts['water'] > 100:
  # If over 100 water violations, publish to water_critical
  sns.publish(
    TopicArn = dept_arns['water_critical'],
    Message = "{} water issues".format(vcounts['water']),
    Subject = "Help fix water violations NOW!")

if vcounts['water'] > 300:
  # If over 300 violations, publish to water_extreme
  sns.publish(
    TopicArn = dept_arns['water_extreme'],
    Message = "{} violations! RUN!".format(vcounts['water']),
    Subject = "THIS IS BAD.  WE ARE FLOODING!")

# Rekognition
    # computer vision API 
    # detecting objects from an image, extracting text from an image

s3 = boto3.client(
    's3', region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET)

s3.upload_file(Filename='report.jpg', Key='report.jpg', Bucket='datacamp-img')

rekog = boto3.client(
    'rekognition', region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET)

response = rekog.detect_labels(
    Image={'S3Object': {
            'Bucket': 'datacamp-img',
            'Name': 'report.jpg'}})

# type of detection can be line or word

# Use Rekognition client to detect labels
image1_response = rekog.detect_labels(
    # Specify the image as an S3Object; Return one label
    Image=image1, MaxLabels=1)

# Print the labels
print(image1_response['Labels'])

# Use Rekognition client to detect labels
image2_response = rekog.detect_labels(
    # Specify the image as an S3Object; Return one label
    Image=image2, MaxLabels=1)

# Print the labels
print(image2_response['Labels'])

# Create an empty counter variable
cats_count = 0
# Iterate over the labels in the response
for label in response['Labels']:
    # Find the cat label, look over the detected instances
    if label['Name'] == 'Cat':
        for instance in label['Instances']:
            # Only count instances with confidence > 85
            if (instance['Confidence'] > 85):
                cats_count += 1
# Print count of cats
print(cats_count)

# Create empty list of words
words = []
# Iterate over the TextDetections in the response dictionary
for text_detection in response['TextDetections']:
  	# If TextDetection type is WORD, append it to words list
    if text_detection['Type'] == 'WORD':
        # Append the detected text
        words.append(text_detection['DetectedText'])
# Print out the words list
print(words)

# Create empty list of lines
lines = []
# Iterate over the TextDetections in the response dictionary
for text_detection in response['TextDetections']:
  	# If TextDetection type is Line, append it to lines list
    if text_detection['Type'] == 'LINE':
        # Append the detected text
        lines.append(text_detection['DetectedText'])
# Print out the words list
print(lines)

translate = boto3.client('translate',
            region_name='us-east-1',
            aws_access_key_id=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET)

response = translate.translate_text(
    Text='Hello, how are you?',
    SourceLanguageCode='auto',
    TargetLanguageCode='es'['TranslatedText'])

comprehend = boto3.client('comprehend',
            region_name='us-east-1',
            aws_access_key_id=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET) 

response = comprehend.detect_dominant_language(
    Text="Hay basura por todas partes a lo loargo de la carretera.")

# Detecting sentiment (can be neutral, positive, negative, mixed)
response = comprehend.detect_sentiment(
    Text="Datacamp students are amazing.", 
    LanguageCode='en')['Sentiment']

# Exercises:
# For each dataframe row
for index, row in dumping_df.iterrows():
    # Get the public description field
    description = dumping_df.loc[index, 'public_description']
    if description != '':
        # Detect language in the field content
        resp = comprehend.detect_dominant_language(Text=description)
        # Assign the top choice language to the lang column.
        dumping_df.loc[index, 'lang'] = resp['Languages'][0]['LanguageCode']
        
# Count the total number of spanish posts
spanish_post_ct = len(dumping_df[dumping_df.lang == 'es'])
# Print the result
print("{} posts in Spanish".format(spanish_post_ct))

for index, row in dumping_df.iterrows():
  	# Get the public_description into a variable
    description = dumping_df.loc[index, 'public_description']
    if description != '':
      	# Translate the public description
        resp = translate.translate_text(
            Text=description, 
            SourceLanguageCode='auto', TargetLanguageCode='en')
        # Store original language in original_lang column
        dumping_df.loc[index, 'original_lang'] = resp['SourceLanguageCode']
        # Store the translation in the translated_desc column
        dumping_df.loc[index, 'translated_desc'] = resp['TranslatedText']
# Preview the resulting DataFrame
dumping_df = dumping_df[['service_request_id', 'original_lang', 'translated_desc']]
dumping_df.head()

for index, row in dumping_df.iterrows():
  	# Get the translated_desc into a variable
    description = dumping_df.loc[index, 'public_description']
    if description != '':
      	# Get the detect_sentiment response
        response = comprehend.detect_sentiment(
          Text=description, 
          LanguageCode='en')
        # Get the sentiment key value into sentiment column
        dumping_df.loc[index, 'sentiment'] = response['Sentiment']
# Preview the dataframe
dumping_df.head()

# Case Study
# Need to initialize boto3 service clients
rekog = boto3.client(
    'rekognition', region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET)

comprehend = boto3.client(
    'comprehend', region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET)

translate = boto3.client(
    'translate', region_name='us-east-1',
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET)

for index, row in df.iterrows():
    desc = df.loc[index, 'public_description']
    if desc != '':
        resp = translate_fake.translate_text(
            Text=desc,
            SourceLanguageCode='auto',
            TargetLanguageCode='en')
        df.loc[index, 'public_description'] = resp['TranslatedText']

for index, row in df.iterrows():
    desc = df.loc[index, 'public_description']
    if desc != '':
        resp = comprehend.detect_sentiment(
           Text=desc,
           LanguageCode='en')
        df.loc[index, 'sentiment'] = resp['Sentiment']

df['img_scooter'] = 0
for index, row in df.iterrows():
    image = df.loc[index, 'image']
    response = rekog.detect_labels(
        Image={'S3Object': {'Bucket': 'gid-images', 'Name': image}}
    )
    for label in response['Labels']:
        if label['Name'] == 'Scooter':
            df.loc[index, 'img_scooter'] = 1
            break

pickups = df[((df.img_scooter == 1) & (df.sentiment == 'NEGATIVE'))]
num_pickups = len(pickups)

# Exercises:
for index, row in scooter_requests.iterrows():
  	# For every DataFrame row
    desc = scooter_requests.loc[index, 'public_description']
    if desc != '':
      	# Detect the dominant language
        resp = comprehend.detect_dominant_language(Text=desc)
        lang_code = resp['Languages'][0]['LanguageCode']
        scooter_requests.loc[index, 'lang'] = lang_code
        # Use the detected language to determine sentiment
        scooter_requests.loc[index, 'sentiment'] = comprehend.detect_sentiment(
          Text=desc, 
          LanguageCode=lang_code)['Sentiment']
# Perform a count of sentiment by group.
counts = scooter_requests.groupby(['sentiment', 'lang']).count()
counts.head()

# Get topic ARN for scooter notifications
topic_arn = sns.create_topic(Name='scooter_notifications')['TopicArn']

for index, row in scooter_requests.iterrows():
    # Check if notification should be sent
    if (row['sentiment'] == 'NEGATIVE') & (row['img_scooter'] == 1):
        # Construct a message to publish to the scooter team.
        message = "Please remove scooter at {}, {}. Description: {}".format(
            row['long'], row['lat'], row['public_description'])

        # Publish the message to the topic!
        sns.publish(TopicArn = topic_arn,
                    Message = message, 
                    Subject = "Scooter Alert")