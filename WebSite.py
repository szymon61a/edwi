import operator


class WebSite():
    __slots__ = 'url', 'mainUrl'

    def parseToRanke(self):
        mList = []
        try:
            for x in range(len(self.url)):
                if mList.__contains__(self.mainUrl) == False:
                    mList.append(self.mainUrl)
                if mList.__contains__(self.url[x].mainUrl) == False:
                    mList.append(self.url[x].mainUrl)
                    mList = mList + self.url[x].parseToRanke()
        except AttributeError:
            return mList
        except RecursionError:
            return mList

        return mList


# def converttonumpy(mainWeb, deapth, counter):
#     mList = []
#     mList.append(mainWeb.mainUrl)
#     try:
#         for web in mainWeb.url:
#             mList = mList + converttonumpy(web, deapth, counter)
#     except AttributeError:
#         return mList
#     return mList

def converttonumpy(mainWeb, endList=None,test=None):
    if (endList == None):
        endList = []
    if (test == None):
        test = []
    if endList.__contains__(mainWeb.mainUrl)==False:
        endList.append(mainWeb.mainUrl)
        test.append(mainWeb)
        tup=(endList,test)
    try:
        for web in mainWeb.url:
            temptup = converttonumpy(web, endList,test)
            if temptup!=None:
                listt=temptup[0]
                testt=temptup[1]
        tup = (endList, test)
    except AttributeError:
        return None
    return tup



def convertnumpy(mainWeb, lista):
    value=[]
    for web in mainWeb:
        webValue=[]
        temp=[]
        try:
            for w in web.url:
                temp.append(w.mainUrl)
            for object in lista:
                    if (temp.__contains__(object)):
                        webValue.append(1)
                    else:
                        webValue.append(0)
            div = len(web.url)
            if(div==0):
                div=1
            webValue = [x / div for x in webValue]
            value.append(webValue)
        except AttributeError:
            temp1 = []
            for x in range(len(lista)):
                temp1.append(1 / len(lista))
            value.append(temp1)
    return value



def printTemp(mList, num):
    dic = {}
    for x in range(len(mList)):
        dic[mList[x]] = num.item(x)
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_x)