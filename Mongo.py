from pymongo import MongoClient

subject="subject"

class Mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.messages
    db.messages.create_index( [ ('subject', 'text')])

    def pushText(self, web, content, rank):
        self.db.messages.insert({"website": web, "subject": content, "rank": str(rank)})

    # def prepare_search(self):


    def search(self, text_f):
        # self.db.messages.createIndex({"content": "text"})
        list=[]
        for cursor in self.db.messages.find({'$text':{"$search":text_f}},{'score':{"$meta":"textScore"}}):
            list.append(cursor)

        for x in reversed(list):
            print(x)

        # cursor
        # for c in cursor:
        #     print(c)