#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 00:57:37 2021

@author: owlthekasra
"""


#!/bin/python

import urllib
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import time
import requests

page_url = "https://wikipedia.org"    
head_url = "/wiki/Wikipedia:Red_Link"
url = page_url + head_url

# Needed help to figure out validation, so I used code 
# from https://github.com/ChrisJamesC/wikipediaPhilosophy/commit/25bc2b491fd3ca08a59ad6d084a98190a2da99f3
# And modified accordingly to only include validation requirements
# I put comments to show that I understood what parts I think relate to the requested validation requirements
def isValid(ref,paragraph):
    if not ref  or "//" in ref:
        #if there are no references, 
        # excludes links in italics, which redirect to a wikipedia page with https://
        # would exclude red links too, but red links written differently inside tag
        return False
    if "/wiki/" not in ref:
        # excludes red links
        return False
    if ref not in paragraph:
        # if link not within a p or ul tag
        # excludes links in boxes (table tag), 
        # excludes footnotes,
        # as well as links that are part of notes, which are in italics
        return False
    prefix = paragraph.split(ref,1)[0]
    if prefix.count("(")!=prefix.count(")"):
        # link not within parentheses
        return False
    return True

# validateTag is completely copied from code above, no modifications
# checks to make sure tag is either a paragraph or unordered list
def validateTag(tag):
    # Check whether the tag is one in which we could find a valid link 
    name = tag.name
    isParagraph = name == "p"
    isList = name == "ul"
    return isParagraph or isList

# technically finds all the valid links within a Wikipedia page,
# but I broke it after the first link to reduce computation load
def findAllLinks(url):
    req = Request(url, headers={'User-Agent' : "Magic Browser"})
    soup = BeautifulSoup(urlopen(req).read())
    soup = soup.find("div", {"class": "mw-parser-output"})
    ref2 = []
    #next part also in code above, I modified it a bit to suit my purposes
    for paragraph in soup.find_all(validateTag, recursive=False):
        for newLink in paragraph.find_all("a"):
            if (len(ref2) == 1):
                break
            ref = newLink.get("href")
            if isValid(str(ref),str(paragraph)):
                ref2.append(ref)
    return ref2

# retrieves and then requests first link recursively until conditions are met
# returns list of first links until 
#   1. there is a repetition, 
#   2. the philosophy page is reached, 
#   3. or there are no valid links
def loopThroughFirstLink(head_url):
    links = []   
    while (head_url != '/wiki/Philosophy'):
        if head_url not in links:
            print(head_url)
            links.append(head_url)
            #put sleep before findAllLinks, 
            #since one request is made in that function
            time.sleep(.5)
            link = findAllLinks(page_url + head_url)
            if len(link) == 0:
                break
            head_url = link[0]
        else:
            break
    print("final: " + head_url)
    links.append(head_url)
    return links

# Gets wiki head of link from the Special:Random request,
# Will not work for cases where link is not title, but I didn't encounter any
def getTitleFromSpecialRandom():
    random = "https://en.wikipedia.org/wiki/Special:Random"
    url = requests.get(random)
    soup = BeautifulSoup(url.content, "html.parser")
    title = soup.find(class_="firstHeading").text.replace(" ", "_")
    full_title = "/wiki/" + title
    return full_title

# Testing
#   Found examples for all three cases, did not find an erroneous example,
#   however, I acknowledge that they could exist
links = loopThroughFirstLink(getTitleFromSpecialRandom())
