1) Add a replset in your /etc/mongod.conf like this:

    replication:
        oplogSizeMB: 128
        replSetName: rs0

2) Connect to your mongodb instance

    $ mongo


2) Initiate the replSet:

     rs.initiate( {
       _id : "rs0",
       members: [ {'_id': 'rs0', 'members': [ {'_id': 0, 'host': 'localhost:27017'}]} ]
    })

3) Find the line
    client = MongoClient(DatabaseSettings.HOST, replicaset='rs0')

    and set the host (probably localhost)

    client = MongoClient('localhost', replicaset='rs0')

4) When Using this class you should use the oplog trigger, unless you know what you are doing ;-)

    Example


    Import the trigger
    -------------------------------
    from Database.Trigger import subscribe_to_collection, Operation


    Usage
    -------------------------------
    # Some user defined function that should be called / action to be performed
    def print_out(doc):
        print (doc)

    # Example: Add a new subscription
    subscribe_to_collection("your_database.your_collection", print_out, Operation.INSERT)

    ---> Your user defined function should always take a mongo doc as a first parameter.
         In this case, every time a new document is inserted, print_out will be called
         with the new doc as the parameter


