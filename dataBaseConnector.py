import pymysql.cursors
from configparser import SafeConfigParser

class dataBaseConnector(object):

    def __init__(self, dbConnectionDetails):

        config = SafeConfigParser()
        config.read(dbConnectionDetails)

        host = config.get('details', 'host')
        user = config.get('details', 'user')
        password = config.get('details', 'password')
        db = config.get('details', 'db')

        self.connection = pymysql.connect(host= host,
                             user= user,
                             password= password,
                             db= db,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor,
                             autocommit = True)

        self.cursor = self.connection.cursor()


    def insert(self, sql):
        try:
            result = self.cursor.execute(sql)
        except Exception as e:
            print(str(e))

    def insert(self, sql, data):
        try:
            result = self.cursor.execute(sql, data)
        except Exception as e:
            print(str(e))

    def execute(self,sql):
        self.cursor.execute(sql)

        data = self.cursor.fetchall()
        return data

