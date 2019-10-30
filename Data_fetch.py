# -*- coding: utf-8 -*-
#coding=utf-8
import sys
import jsonpickle
import os
import tweepy
import time
import datetime
from datetime import timedelta
import re
import codecs
import xlrd
from xlrd import open_workbook
import preprocessor as p
import csv 

#Twitter initialization


consumer_key = 'sthBjkpMVS2Iaw9rW7KmY19Om'
consumer_secret = 'drkg7nHMGWMfMG3TvLdmyHkgaECojlQ4L4VOPVp1x1SEvzyCwe'
access_token = '992086107083829248-YBrowo9XyvzyjBnHAd5alXN8eVTn4hP'
access_secret = 'XX9Ruvvo2Xdkv8ED3fhPtzAoBJCLReGHrThcx7HAoGzHm'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
api = tweepy.API(auth)

file_exists = os.path.isfile('C:\Python27\PL.csv')

csvFile = open('PL.csv', 'ab')
fields = ('Tweet_Id', 'Tweet_Text','Tweet_authorscreen_name','Tweet_author_id','Tweet_created_at','Tweet_coordinates','Tweet_source','Tweet_user_verified','Tweet_retweet_count','Tweet_lang','Tweet_favcount','Tweet_username','Tweet_userid','Tweet_location') #field names
csvWriter = csv.DictWriter(csvFile, fieldnames=fields)
if not file_exists:
    csvWriter.writeheader()
    
c = tweepy.Cursor(api.search, q="#love -filter:retweets", since="2019-02-02", until="2019-02-03", lang="en", tweet_mode="extended").items()


count=0;
while True:

          try:
		  
               tweet = c.next()  			   
               print (tweet.id_str, (tweet.full_text.encode('utf-8').replace('\n', '').replace('\r', ' ').decode('unicode_escape').encode('ascii','ignore').strip()), tweet.author.screen_name, tweet.author.id, tweet.created_at,tweet.coordinates,tweet.source,tweet.user.verified,tweet.retweet_count,tweet.lang,tweet.user.favourites_count,tweet.user.name,tweet.user.id_str,tweet.user.location)  
               csvWriter.writerow({'Tweet_Id': tweet.id_str, 'Tweet_Text': (tweet.full_text.encode('utf-8').replace('\n', '').replace('\r', ' ').decode('unicode_escape').encode('ascii','ignore').strip()),'Tweet_authorscreen_name':tweet.author.screen_name.encode('utf-8').strip(),'Tweet_author_id':tweet.author.id,'Tweet_created_at':tweet.created_at,'Tweet_coordinates':tweet.coordinates,'Tweet_source':tweet.source.encode('utf-8').strip(),'Tweet_user_verified':tweet.user.verified,'Tweet_retweet_count':tweet.retweet_count,'Tweet_lang':tweet.lang.encode('utf-8').strip(),'Tweet_favcount':tweet.user.favourites_count,'Tweet_username':tweet.user.name.encode('utf8').strip(),'Tweet_userid':tweet.user.id_str,'Tweet_location':tweet.user.location.encode('utf-8').strip()})
               count +=1
               
          except tweepy.TweepError:
                print("Whoops, could not fetch more! just wait for 15 minutes :")
                time.sleep(900)
                continue
          except StopIteration:
                break
csvFile.close()
print count

