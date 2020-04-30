#!/usr/bin/python3

from pymongo import MongoClient
import datetime

client = MongoClient()
db = client['test-db']

myContent0 = """Here begins a small bit of blather. To blather is to talk on and on without saying anything very important or wise. If you blather all afternoon, it might be a welcome distraction to your friend who's grieving the death of her cat.

You can use the word blather as a noun too: you might hate riding the bus home from school because of all the silly blather around you. At a job, it might be acceptable to blather during your lunch break, but not once you get back to work. The verb came first, and it was originally Scottish, probably from the Old Norse word blaðra, "mutter or wag the tongue."""

myDoc = {'doc': 'doc0',
         'author': 'Mark',
         'tags': ['mongodb', 'python', 'pymongo'],
         'datetime': datetime.datetime.utcnow(),
         'content': myContent0
        }
docs = db.docs
docID = docs.insert_one(myDoc).inserted_id
print(f"docID: {docID}")
print(f"collection names: {db.list_collection_names()}")
