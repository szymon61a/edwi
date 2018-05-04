import collections
import http
import re
import socket
import sys
from urllib import request, error
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
# import requests
# from HtmlParser import ParserHTML
from bs4 import BeautifulSoup
from pattern3.web import URL, HTTP404NotFound, URLError
from scipy import spatial
from scipy.sparse import csc_matrix

from WebSite import WebSite, converttonumpy, convertnumpy, printTemp


def downloadWebSite(url):
    try:
        page = request.urlopen(url)
        html = page.read()
        return html
    except AttributeError:
        print("Error url", url)
    except request.HTTPError:
        return ""


def getTextWebsite(page):
    soup = BeautifulSoup(page, 'html.parser')
    [s.extract() for s in soup(['style', '[document]', 'script', 'head', 'title'])]
    text = re.sub(r'[\s\t\n\r@.\-,.@#$%^&*()+/<>?!`:;0-9\']', ' ', soup.text)
    test = re.findall(r'\b[^\d\W]+\b', soup.text)
    dic = {i: test.count(i) for i in test}
    sort_dic = [(k, dic[k]) for k in sorted(dic, key=dic.get, reverse=True)]
    return sort_dic


def desired_links(href):
    ignore_values = {"", "/", }
    if not href:
        return False
    if href.startswith("#"):
        return False
    if href.startswith("javascript:pop"):
        return False
    if href in ignore_values:
        return False
    return True


def getULRsWebsite(page):
    soup = BeautifulSoup(page, 'html.parser')
    urls = [s.get('href') for s in soup.find_all('a')]
    return urls


def getScriptsWebsite(page):
    soup = BeautifulSoup(page, 'html.parser')
    urls = [s.get('src') for s in soup.find_all('script')]
    text = list(filter(None, urls))
    return text


def getImagesWebsite(page):
    soup = BeautifulSoup(page, 'html.parser')
    urls = [s.get('src') for s in soup.find_all('img')]
    return urls


def save_to_file(file, text):
    file = open(file, 'w')
    file.write(text)


def thissameSystem(mUrl, depth):
    prefix = mUrl.split('//')
    mUrl = URL(mUrl)
    domain = mUrl.domain
    page = mUrl.read()
    listPlay = []
    listEnd = []
    listNext = []
    soup = BeautifulSoup(page, 'html.parser')
    listPlay = [s.get('href') for s in soup.find_all('a', href=desired_links)]
    for url in listPlay:
        t = urlparse(url)
        temp_domain = t.netloc
        if (domain == temp_domain):
            listNext.append(url)
        if (url[-4::] == "html" and '' == temp_domain):
            listNext.append(prefix[0] + '//' + domain + '/' + url)
    listPlay = listNext
    listNext = []
    for a in range(depth):
        for url in listPlay:
            if (listEnd.__contains__(url) == False):
                try:
                    listEnd.append(url)
                    temp_web = urlparse(url).netloc
                    print(url)
                    temp_page = downloadWebSite(url)
                    soup = BeautifulSoup(temp_page, 'html.parser')
                    list = [s.get('href') for s in soup.find_all('a', href=desired_links)]
                    for temp_url in list:
                        # print(temp_web.domain,temp_url)
                        try:
                            tt = urlparse(temp_url)
                            temp_domain = tt.netloc
                            if (temp_web == temp_domain):
                                text = url
                                if (listNext.__contains__(text) == False and listEnd.__contains__(text) == False):
                                    listNext.append(text)
                            if (temp_url[-4::] == "html" and '' == temp_domain):
                                text = prefix[0] + '//' + temp_web + '/' + temp_url
                                if (listNext.__contains__(text) == False and listEnd.__contains__(text) == False):
                                    listNext.append(text)
                        except error.HTTPError:
                            print("error2")
                        except socket.timeout:
                            print("socket error2")
                except error.HTTPError:
                    print("error")
                except socket.timeout:
                    print("socket error")
        listPlay = listNext
        listNext = []

    print(len(listEnd))
    return listEnd


def outsideurl(mainweb, deapth, counter):
    try:
        print(mainweb.mainUrl)
        mUrl = URL(mainweb.mainUrl)
        page = mUrl.read()
        soup = BeautifulSoup(page, 'html.parser')
        lists = [s.get('href') for s in soup.find_all('a', href=desired_links)]
        mainweb.url = []
        for param in lists:
            try:
                if (urlparse(mainweb.mainUrl).netloc != urlparse(param).netloc and "" != urlparse(param).netloc):
                    web = WebSite()
                    web.mainUrl = param
                    mainweb.url.append(web)
                    if (deapth != counter):
                        counter = counter + 1
                        outsideurl(web, deapth, counter)
            except HTTP404NotFound:
                print("error")
    except HTTP404NotFound:
        print("error : ", mainweb.mainUrl)
    except URLError:
        print('error : ', mainweb.mainUrl)
    except socket.timeout:
        print('time out', mainweb.mainUrl)
    except http.client.IncompleteRead:
        print("IncompleteRead : ", mainweb.mainUrl)


def allfunction(flags, url, flag=None):
    page = downloadWebSite(url)
    text = ""
    urls = []
    for y in range(len(flags[::])):
        if (flags[y] == "-text"):
            text += str(getTextWebsite(page))
            text += "\n"

        if (flags[y] == "-a"):
            text += str(getULRsWebsite(page))
            text += "\n"

        if (flags[y] == "-script"):
            text += str(getScriptsWebsite(page))
            text += "\n"

        if (flags[y] == "-img"):
            text += str(getImagesWebsite(page))
            text += "\n"
        if (flags[y] == "-graph"):
            mainWeb = WebSite()
            mainWeb.mainUrl = url
            outsideurl(mainWeb, int(flags[y + 1]), 0)
            printGraph(mainWeb, int(flags[y + 1]), 0)
            allfunction(flags[y + 1::], url, flag)

        if (flags[y] == "-pr"):
            mainWeb = WebSite()
            mainWeb.mainUrl = url
            outsideurl(mainWeb, int(flags[y + 1]), 0)
            array = converttonumpy(mainWeb)
            # test = collections.Counter(array)
            object = convertnumpy(array[1], array[0])
            numpy = np.array(object)
            # numpy=np.transpose(numpy)
            rank=executeComputations(numpy)
            # rank=pageRank(numpy, s=.86)
            printTemp(array[0], rank)
            allfunction(flags[y + 1::], url, flag)

        if (flags[y] == "-deapth"):

            print(flags[y])
            urls = thissameSystem(url, int(flags[y + 1]))
            for mUrl in urls:
                texts = ""
                print(mUrl)
                texts += allfunction(flags[y + 1::], mUrl, flag)
                text += mUrl + "\n"
                text += texts + "\n"
                # text=text.encode('utf8')
        if (flags[y] == "-cos"):
            for x in urls:
                for y in urls:
                    tab = [x, y]
                    print(tab)
                    print(cos(tab))

    return text


def executeComputations( M):
    damping = 0.80
    error = 0.0000001
    N = M.shape[0]
    v = np.ones(N)
    v = v / np.linalg.norm(v, 1)
    last_v = np.full(N, np.finfo(float).max)
    for i in range(0, N):
        try:
            temp=M[:,i]
            if sum(temp) == 0:
                M[:, i] = np.full(N, 1.0 / N)
        except IndexError:
            print(temp[:])
    M_hat = np.multiply(M, damping) + np.full((N, N), (1 - damping) / N)

    while np.linalg.norm(v - last_v) > error:
        last_v = v
        v = np.matmul(M_hat, v)

    return np.round(v, 6)

def do(mainWeb, deapth, counter):
    mList = []
    if (deapth + 1 != counter):
        counter = counter + 1
        try:
            for x in range(len(mainWeb.url)):
                t = (mainWeb.mainUrl, mainWeb.url[x].mainUrl, counter - 1, counter)
                mList.append(t)
                mList = mList + do(mainWeb.url[x], deapth, counter)
        except AttributeError:
            error = ''
    return mList


def printGraph(mainWeb, deapth, counter):
    mList = do(mainWeb, deapth, counter)
    mFrom = []
    mTo = []
    for object in mList:
        mFrom.append(object[0])
        mTo.append(object[1])

    df = pandas.DataFrame({'from': mFrom, 'to': mTo})
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    # nx.draw(G,with_labels=True,node_size=1000,alpha=0.3,node_color="skyblue",arrows=True,pos=nx.spectral_layout(G))
    # plt.show()
    G1 = nx.from_pandas_edgelist(df, 'from', 'to')
    # pos = nx.spring_layout(G1, iterations=1000)
    # pos = nx.spectral_layout(G1)
    pos = nx.spring_layout(G1)
    nx.draw_networkx(G1, pos, node_size=200, alpha=0.4, edge_color='r', font_size=8, with_labels=True, arrows=True,)
    plt.show()


def cos(urls):
    text = []
    for url in urls:
        text.append(getTextWebsite(downloadWebSite(url)))

    A = []
    B = []
    flags = False
    for a in text[0]:
        A.append(int(a[1]))
        for b in range(len(text[1])):
            if (a[0] == text[1][b][0]):
                B.append(int(text[1][b][1]))
                text[1].pop(b)
                flags = True
                break
        if (flags == False):
            B.append(0)
        flags = False
    for a in text[1]:
        B.append(a[1])
        A.append(0)

    return 1 - spatial.distance.cosine(A, B)


def pageRank(G, s=.85, maxerr=.0001):
    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G, dtype=np.float)
    rsums = np.array(A.sum(1))[:, 0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums == 0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r - ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0, n):  # range
            # inlinks of state i
            Ai = np.array(A[:, i].todense())[:, 0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))

    # return normalized pagerank
    return r / float(sum(r))


def main(arg):
    count = 0
    page = ""

    if (arg[count] == "-url" or arg[count] == "-site"):
        count += 1
        urls = arg[count]
        urls = urls.split(',')
        count += 1
        if (arg[count] == "-file"):
            file = arg[count + 1]
            count += 2
            text = ""
            for url in urls:
                print(url)
                text += allfunction(arg[count::], url, arg)
                text += "\n"
            print(text)
            if (arg[count] == "-cos"):
                text = str(cos(urls))

            save_to_file(file, text)

        if (arg[count] == "-console"):
            count += 1
            for url in urls:
                print(url)
                print(allfunction(arg[count::], url, arg))


# if(arg[count]=="-file"):
#     print()


if __name__ == "__main__":
    main(sys.argv[1:])
