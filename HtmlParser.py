from html.parser import HTMLParser
import re

class ParserHTML(HTMLParser):

    def handle_data(self, data):
        text=str(data)
        text=text.rstrip().split(" ")
        print(text)
        if '' not in text:
            print(text)