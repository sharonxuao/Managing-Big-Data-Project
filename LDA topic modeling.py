# load pyspark with bigquery connector
pyspark --jars=gs://hadoop-lib/bigquery/bigquery-connector-hadoop2-latest.jar

# import libraries
from __future__ import absolute_import
import json
import pprint
import subprocess
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType
from pyspark.sql.functions import struct
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA

# connect to spark session
sc = pyspark.SparkContext()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the InputFormat. This assumes the Cloud Storage connector for
# Hadoop is configured.
bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')

# Set an input directory for reading data from Bigquery.
input_directory = 'gs://{}/hadoop/tmp/bigquery/questions'.format(bucket)

# customize conf with desired table in bigquery
conf = {
    # Input Parameters.
    'mapred.bq.project.id': project,
    'mapred.bq.gcs.bucket': bucket,
    'mapred.bq.temp.gcs.path': input_directory,
    'mapred.bq.input.project.id': project,
    'mapred.bq.input.dataset.id': 'posts_questions',
    'mapred.bq.input.table.id': 'questions',
}

# update names
# Output Parameters.
output_dataset = 'stackoverflow_db'
output_table = 'posts'

# update data
# Load data in from BigQuery.
posts_data = sc.newAPIHadoopRDD(
    'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
    'org.apache.hadoop.io.LongWritable',
    'com.google.gson.JsonObject',
    conf=conf)

# Extract the JSON strings from the RDD.
table_json = posts_data.map(lambda x: x[1])w

# Load the JSON strings as a Spark Dataframe.
posts = spark.read.json(table_json)

#------------------------------------------------------
# import files from bucket
path = "gs://dataproc-7f0d81cb-9a70-4c97-8e3d-c20b659bf779-us-east1/hadoop/tmp/bigquery/questions"

df = spark.read.json(path)

#------------------------------------------------------
# Create a view so that Spark SQL queries can be run against the data.
df.createOrReplaceTempView("table1")

# split tags
df_split = spark.sql("SELECT title, body, tags, split(tags, '\\\|')[0] as tag1, split(tags, '\\\|')[1] as tag2, split(tags, '\\\|')[2] as tag3, split(tags, '\\\|')[3] as tag4, split(tags, '\\\|')[4] as tag5 FROM table1")

#-----------------------------------------------------------------------------------------------------
# spark text analytics on Python topics

# subset posts questions tagged as python
python = spark.sql("SELECT tags, body, title FROM table1 WHERE tags LIKE '%python%' AND creation_date LIKE '2018%'")

# concat title and body
python = spark.sql("SELECT CONCAT(tags, body) AS text FROM table1 WHERE tags LIKE '%python%' AND creation_date LIKE '2018%'")
python.count() # 242,708 records

# define words to be used
def cleanup_text(record):
    import re
    text  = record[0]
    words = text.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves',u'way']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = [u'python', u'error', u'use', u'print', u'elif', u'true', u'false', u'except', u'even', u'else', u'try', u'def', u'gtgtgt', u'need', u'new', u'values',u'<p>',
    u'info', u'like', u'codepre',u'event',u'file',u'code',u'line',u'<p>',u'using',u'get',u'following',u'yes',u'want', u'ltdiv', u'ltdivgt', u'import', u'image']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

# define the function for generating tokenized word list
udf_cleantext = udf(cleanup_text, ArrayType(StringType()))

# clean the text
clean_text = python.withColumn("words", udf_cleantext(struct([python[x] for x in python.columns])))

#from pyspark.mllib.feature import HashingTF - Option 1 # didn't use
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
#featurizedData = hashingTF.transform(clean_text)
 
# Term Frequency Vectorization  - Option 2 (CountVectorizer, choose to use becuse working on dataframes with Spark ML are more effecient): 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 10000, minDF = 2)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

# define bag of words to be used 
vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

# calculate TF-IDF matrix
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData) # TFIDF

# Generate 5 Data-Driven Topics:
# "em" = expectation-maximization 
lda = LDA(k=5, seed=123, optimizer="em", featuresCol="features")
ldamodel = lda.fit(rescaledData)
 
# describe topics
ldatopics = ldamodel.describeTopics()

# Show the top Topics
ldatopics.show()

# Given a vocabulary as a list of words we can index into it to visualize topics：
topics_rdd = ldatopics.rdd # transform back to rdd

topics_words = topics_rdd\
       .map(lambda row: row['termIndices'])\
       .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
       .collect()

for idx, topic in enumerate(topics_words):
    print "topic: ", idx
    print "----------"
    for word in topic:
       print word
    print "----------"

#-----------------------------------------------------------------------------------------
# text analytics on R topics

# subset table
df.createOrReplaceTempView("table1")
df_split = spark.sql("SELECT title, body, creation_date, tags, split(tags, '\\\|')[0] as tag1, split(tags, '\\\|')[1] as tag2, split(tags, '\\\|')[2] as tag3, split(tags, '\\\|')[3] as tag4, split(tags, '\\\|')[4] as tag5 FROM table1")
df_r = spark.sql("SELECT CONCAT(title, body) AS text FROM table1 WHERE creation_date LIKE '2018%' AND (tag1=='r' OR tag2=='r' OR tag3=='r' OR tag4=='r' OR tag5=='r')")

df_r.count() #54431

def cleanup_text(record):
    import re
    text  = record[0]
    words = text.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves',u'way']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = [u'error', u'use', u'print', u'elif', u'true', u'false', u'except', u'even', u'else', u'try', u'def', u'gtgtgt', u'need', u'new', u'values',u'<p>',
    u'info',u'codepre',u'event',u'file',u'code',u'line',u'item',u'data',u'using',u'like',u'yes',u'ltdblgt',u'noreferrerimg',u'relnofollow', u'good',u'ltnagt',u'running',u'one',u'html',u'two',u'male',u'blockquote']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))

# clean the text
clean_text = python.withColumn("words", udf_cleantext(struct([python[x] for x in python.columns])))

# Term Frequency Vectorization  - Option 2 (CountVectorizer, choose to use becuse working on dataframes with Spark ML are more effecient): 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 10000, minDF = 2)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

# define bag of words to be used 
vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

# calculate TF-IDF matrix
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData) # TFIDF

# Generate 5 Data-Driven Topics:
# "em" = expectation-maximization 
lda = LDA(k=5, seed=123, optimizer="em", featuresCol="features")
ldamodel = lda.fit(rescaledData)
 
# describe topics
ldatopics = ldamodel.describeTopics()

# Show the top Topics
ldatopics.show()

# Given a vocabulary as a list of words we can index into it to visualize topics：
topics_rdd = ldatopics.rdd # transform back to rdd

topics_words = topics_rdd\
       .map(lambda row: row['termIndices'])\
       .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
       .collect()

for idx, topic in enumerate(topics_words):
    print "topic: ", idx
    print "----------"
    for word in topic:
       print word
    print "----------"

#-----------------------------------------------------------------------------------------
# work with tags
def create_tag_frequencies(dataframe):
        """Produces a PySpark dataframe containing a column representing the total frequency of the tags by record.
        The frequency of tags is determined by their proportion of the total number of tags in the dataframe.
        :param dataframe: the PySpark dataframe
        :returns: the PySpark dataframe containing the tag frequency field and all fields in the supplied dataframe
        """
        from pyspark.sql.functions import desc
        from pyspark.sql.functions import col
        df_tags = dataframe.selectExpr("tag1 AS tag").union(dataframe.selectExpr("tag2 AS tag")).union(dataframe.selectExpr("tag3 AS tag")) \
                           .union(dataframe.selectExpr("tag4 AS tag")).union(dataframe.selectExpr("tag5 AS tag"))
        df_tags = df_tags.na.drop(subset=["tag"])
        tags_total_count = df_tags.count()
        print("Total number of tags used, including duplicates:",tags_total_count)
        df_tag_freq = df_tags.groupBy("tag").count().orderBy(desc("count"))
        df_tag_freq = df_tag_freq.withColumn("frequency", col("count")/tags_total_count)
        df_tag_freq.orderBy(desc("frequency")).show(20)

create_tag_frequencies(df)