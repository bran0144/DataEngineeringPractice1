# Three V's of Big Data
    # Volume - size of the data
    # Variety - different sources and formats
    # Velocity - speed that data is generated and available for processing

# Clustered computing - collection of resources of multiple machines
# Parallel computing - type of computation carried out simultaneously
# Distributed computing - nodes or networked computers that run jobs in parallel
# Batch processing - breaking data into smaller pieces and running each piece on an
    # individual machine
# Real time processing - demands that information is processed and made ready 
    # immediately

# Processing Systems
    # Hadoop/MapReduce - scalable fault tolerant written in Java
        # open source, good for batch processing
    # Apache Spark - general purpose lighnting fast cluster computing system
        #framework for storing and processing across clustered computers
        # open source, good for batch processing and real time data processing

# Apache Spark
    # distributes data and computation across a distributed cluster
    # executes complex multi-stage applications such as ML
    # efficient in-memory computations for large data sets
    # Very fast
    # Spark is written in Scala, but supports Java, Python, R and SQL
    # Spark Core - contains the basic functionality of Spark with libraries built on top
#Spark Libraries
    # Spark SQL - processes structured and semi-structured data in Python, Java, and Scala
    # MLib - library of common machine learning algorithms
    # GraphX - collection of algorithms for manipulating graphs and performing parallel graph computations
    # Spark Streaming - scalable, high throughput processing library for real time data
# Spark Modes
    # Local mode - single machine (like your laptop)
        # convenient for testing, debugging, and demonstration
    # Cluster mode - set of predefined machines
        # mainly used for production
    # Typical workflow is that you start in local and transition to cluster - no code change necessary

#PySpark
