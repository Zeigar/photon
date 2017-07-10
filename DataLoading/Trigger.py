import pymongo
import time

from pymongo import MongoClient, CursorType
from threading import Thread

from Database import DatabaseSettings
from Logging import Logger

""" The oplog_trigger, tail_trigger and interval_puller  allow you to add trigger functions.
 
   
    Generally
    ----------------------------------------------------------------------------------------------
    You should use the tail_trigger whenever possible because it is should be more efficient.
    It does not work with non-capped collections! Use interval_puller sparsely. It's bad and
    you should feel bad using it.
 
 
    oplog_trigger:
    ----------------------------------------------------------------------------------------------
    The oplog_trigger observes mongo's oplog and allows a callback to take place. You should use 
    this when observing a non-capped-collection (and you can use it for a capped-collection if you 
    wish to do so). Configure a mongo replSet before using this (nescessary for oplog observing).
    
    
    tail_trigger:
    ----------------------------------------------------------------------------------------------
    The tail_trigger only works on capped collections. The collection must be capped,
    because a normal collection does not allow oplog tailing.
    
    A capped collection must be created beforehand like

        collection = db.create_collection('trigger_testing', capped=True, size=5242880, max=5000)

    The given user function must take as a first argument the doc that was added to the collection.
    
    
    interval_puller:
    ----------------------------------------------------------------------------------------------
    interval_pulling regularly pulls the given timestamp field of the given collection. Whenever
    a document with a timestamp greater than the last pulled timestamp (with more than one result).
    
    The given call back function can stop the execution by setting the should_continue param to
    false.
    
    Make sure, that the queried timestamp is an indexed field! Otherwise the performance might 
    not be acceptable long-term (especially with only small sleeping periods).
    
"""

# Unitialized connect
# client = MongoClient("localhost:27017")
# config = {'_id': 'rs0', 'members': [ {'_id': 0, 'host': 'localhost:27017'}]}
# client.admin.command("replSetInitiate", config)

# Standard Connect

client = MongoClient(DatabaseSettings.HOST, replicaset='rs0')

# Contains all collection subscriptions
oplog_subscribers = []



class Operation:
    """ 
        The oplog trigger will be called whenever a document is inserted,
        updated or deleted. It is possible to subscribe to only one of those
        types.
    """
    INSERT = 'i'
    UPDATE = 'u'
    DELETE = 'd'
    ALL = 'a'



def subscribe_to_collection(collection, func_callback, type = Operation.ALL):
    """ 
        Add a function callback to a collection. 
        func_callback will be called with the new doc when it is registered in the oplog. 
        You can define which kind of operation you want to be notified about by
        setting the type parameter.
    """
    oplog_subscribers.append({'collection': collection,  'func_callback': func_callback, 'type': type})
    Logger.info('Subscribed ' +str(func_callback) + ' to ' + collection + '(Type: '+str(type)+')')


def _doc_to_subscribers(doc):
    """ Sends the doc to the belonging subscriber functions """

    # This is not optimal yet (but good enough, for now)
    for subscriber in oplog_subscribers:
        if(doc['ns'] == subscriber['collection'] and
               (subscriber['type'] == Operation.ALL or doc['op'] == subscriber['type'])):
                    subscriber['func_callback'](doc)



def _oplog_trigger():
    """ oplog_trigger observes mongo's oplog (must be activated for the db!)"""
    Logger.info('oplog_trigger initialized!')

    oplog = client.local.oplog.rs
    start = oplog.find().sort('$natural', pymongo.DESCENDING).limit(-1).next()
    ts = start['ts']

    while True:
        cursor = oplog.find({'ts': {'$gt': ts}}, cursor_type = CursorType.TAILABLE_AWAIT)
        while cursor.alive:
            for doc in cursor:
                _doc_to_subscribers(doc)

            time.sleep(1)


def _tail_trigger(capped_collection, user_function, *argv):
    """ Observes the tail of a capped collection. """

    cursor = capped_collection.find(cursor_type = CursorType.TAILABLE_AWAIT)

    while cursor.alive:
        try:
            doc = cursor.next()
            user_function(doc, *argv)

        except StopIteration:
            time.sleep(1)


def _interval_puller(collection, timestamp_field: str, sleep: int, callback, ts = 0, should_continue:bool = True):
    """ Pulls data in intervals """

    new_ts = time.time() * 1000
    count = collection.find({timestamp_field: {'$gt': ts}}).count()

    if count > 0:
        should_continue = callback(ts)

    if should_continue:
        time.sleep(sleep)
        _interval_puller(collection, timestamp_field, sleep, callback, new_ts, should_continue)



""""    
    Public functions that should be called to initialize an observer / trigger. 
    
    Using these functions makes sure, that the triggers are started in a new thread (and thus are
    non-blocking). You should not use the private functions!
"""

def start_oplog_trigger():
    """ 
        You should use this whenever you want to add a callback to a non-capped collection. 
        After initializing the oplog_trigger you can add collections and callbacks by
        using
    
            subscribe_to_collection("photon_log_db.info_log", callback_function)
    """
    t = Thread(target=_oplog_trigger)
    t.setDaemon(True)
    t.start()


def start_interval_puller(collection, timestamp_field: str, sleep: int, callback, ts = 0, should_continue:bool = True):
    """ 
        You should not use this, unless everything else does not work in your case or your use-case is
        dependent on using interval pulling
    """
    t = Thread(target=_interval_puller, args=(collection, timestamp_field, sleep, callback, ts, should_continue))
    t.setDaemon(True)
    t.start()


def start_tail_trigger(capped_collection, user_function, *argv):
    """
        Use this on capped collections, if you don't want to use the oplog trigger. This one may be a bit
        faster than oplog observing
    """
    t = Thread(target=_tail_trigger, args=(capped_collection, user_function, argv))
    t.setDaemon(True)
    t.start()


# Must be called once to start observer Thread
# This call should (must?!) be made only once in our program!
# We should think hard about where to put
start_oplog_trigger()

if __name__ == "__main__":

    # User defined function 1 -> tail_trigger
    def trig_func(doc: dict):
        print('TRIGGER: ' + str(doc['val']))

    # User defined function 2 -> interval_puller
    def get_new_docs(ts):
        count = collection.find({'created_at': {'$gt': ts}}).count()
        return True # should continue!

    # User defined function 2 -> oplog trigger
    def print_out(doc):
        print (doc)


    # Example subscription
    subscribe_to_collection('trigger_db.trigger_testing', print_out, Operation.ALL)

    # Test-Database
    db = client.trigger_db

    # Must be a capped collection!
    collection = db.trigger_testing

    # Initialize the tail trigger
    start_tail_trigger(collection, trig_func, 1, 2)

    # Initialize the interval puller
    # start_interval_puller(collection, 'created_at', 3,  get_new_docs, 0)

    i = 1
    while i < 25:
        collection.insert_one({'val': i, 'created_at': time.time() * 1000})
        i+=1
