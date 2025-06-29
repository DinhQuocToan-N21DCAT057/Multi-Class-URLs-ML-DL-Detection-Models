import whois
from datetime import datetime, timezone
import math
import pandas as pd
import numpy as np
from pyquery import PyQuery
from requests import get


class URL_EXTRACTOR(object):
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.today = datetime.now().replace(tzinfo=None)

        try:
            self.whois = whois.query(self.domain).__dict__
        except:
            self.whois = None
        
        try:
            self.response = get(self.url)
            self.pq = PyQuery(self.response.text) 
        except:
            self.response = None
            self.pq = None

    ########################################################################################
    #  ___     _   _ ____  _       ____ _____ ____  ___ _   _  ____                        #
    # |_ _|   | | | |  _ \| |     / ___|_   _|  _ \|_ _| \ | |/ ___|                       #
    #  | |    | | | | |_) | |     \___ \ | | | |_) || ||  \| | |  _                        #
    #  | | _  | |_| |  _ <| |___   ___) || | |  _ < | || |\  | |_| |                       #
    # |___(_)__\___/|_| \_\_____| |____/_|_|_|_| \_\___|_| \_|\____|                       #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                     #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                     #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                    #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                     #
    ########################################################################################

    #######################################################################################
    #                            1.1 Entropy of URL                                       #
    #######################################################################################
    
    def entropy(self):
        str = self.url.strip()
        prob = [float(str.count(c)) / len(str) for c in dict.fromkeys(list(str))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy
    
    #######################################################################################
    #                           1.2 Occurence of IP in URL                                #
    #######################################################################################

    def has_ip(self):
        str = self.url
        flag = False
        if ("." in str):
            elements_arr = str.strip().split(".")
            if (len(elements_arr) == 4):
                for i in elements_arr:
                    if (i.isnumeric() and int(i) >= 0 and int(i) <= 255):
                        flag = True
                    else:
                        flag = False
                        break
        if flag: return 1
        else: return 0

    #######################################################################################
    #                     1.3 Total number of digits in URL string                        #
    #######################################################################################
    
    def numDigits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    #######################################################################################
    #                    1.4 Total number of characters in URL string                     #
    #######################################################################################

    def urlLength(self):
        return len(self.url)
    
    #######################################################################################
    #                    1.5 Total number of query parameters in URL                      #
    #######################################################################################

    def numParameters(self):
        params = self.url.split('&')
        return len(params) - 1
    
    #######################################################################################
    #                         1.6 Total Number of Fragments in URL                        #
    #######################################################################################

    def numFragments(self):
        fragments = self.url.split('#')
        return len(fragments) - 1
    
    ########################################################################################
    #  ___ ___     _   _ ____  _       ____   ___  __  __    _    ___ _   _                #
    # |_ _|_ _|   | | | |  _ \| |     |  _ \ / _ \|  \/  |  / \  |_ _| \ | |               #
    #  | | | |    | | | | |_) | |     | | | | | | | |\/| | / _ \  | ||  \| |               #
    #  | | | | _  | |_| |  _ <| |___  | |_| | |_| | |  | |/ ___ \ | || |\  |               #
    # |___|___(_)_ \___/|_|_\_\_____|_|____/_\___/|_|  |_/_/   \_\___|_| \_|               #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                     #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                     #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                    #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                     #    
    ########################################################################################

    #######################################################################################
    #                           2.1 Domain has http protocol                              #
    #######################################################################################

    def hasHttp(self):
        return 'http:' in self.url
    
    #######################################################################################
    #                           2.2 Domain has https protocol                             #
    #######################################################################################

    def hasHttps(self):
        return 'https:' in self.url
    
    #######################################################################################
    #                 2.3 Number of days from today since domain was registered           #
    #######################################################################################

    def daysSinceRegistration(self):
        if self.whois and self.whois['creation_date']:
            diff = self.today - self.whois['creation_date'].replace(tzinfo=None)
            diff = str(diff).split(' days')[0]
            return diff
        else: return 0

    #######################################################################################
    #                  2.4 Number of days from today since domain expired                 #
    #######################################################################################

    def daysSinceExpiration(self):
        if self.whois and self.whois['expiration_date']:
            diff = self.whois['expiration_date'].replace(tzinfo=None) - self.today
            diff = str(diff).split(' days')[0]
            return diff
        else: return 0

    #######################################################################################
    #  ___ ___ ___     _   _ ____  _       ____   _    ____ _____                         #
    # |_ _|_ _|_ _|   | | | |  _ \| |     |  _ \ / \  / ___| ____|                        #
    #  | | | | | |    | | | | |_) | |     | |_) / _ \| |  _|  _|                          #
    #  | | | | | | _  | |_| |  _ <| |___  |  __/ ___ \ |_| | |___                         #
    # |___|___|___(_) _\___/|_|_\_\_____| |_|_/_/___\_\____|_____|                        #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                  3.1 Total number of characters in URL's HTML page                  #
    #######################################################################################

    def bodyLength(self):
        if self.pq is not None:
            return len(self.pq('html').text()) if self.urlIsLive else 0
        else: return 0

    #######################################################################################
    #                  3.2 Total number of HI-H6 titles in URL's HTML page                #
    #######################################################################################

    def numTitles(self):
        if self.pq is not None:
            titles = ['h{}'.format(i) for i in range(7)]
            titles = [self.pq(i).items() for i in titles]
            return len([item for s in titles for item in s])
        else: return 0
    
    #######################################################################################
    #               3.3 Total number of images embedded in URL's HTML page                #
    #######################################################################################

    def numImages(self):
        if self.pq is not None:
            return len([i for i in self.pq('img').items()])
        else: return 0

    #######################################################################################
    #               3.4 Total number of links embedded in URL's HTML page                 #
    #######################################################################################

    def numLinks(self):
        if self.pq is not None:
            return len([i for i in self.pq('a').items()])
        else: return 0

    #######################################################################################
    #        3.5 Total number of characters in embedded scripts in URL's HTML page        #
    #######################################################################################

    def scriptLength(self):
        if self.pq is not None:
            return len(self.pq('script').text())
        else: return 0
    
    #######################################################################################
    #               3.6 Total number of special characters in URL's HTML page             #
    #######################################################################################

    def specialCharacters(self):
        if self.pq is not None:
            bodyText = self.pq('html').text()
            schars = [i for i in bodyText if not i.isdigit() and not i.isalpha()]
            return len(schars)
        else: return 0

    #######################################################################################
    #                       3.7 The ratio of total length of embedded                     #
    #                       scripts to special characters in HTML page                    #
    #######################################################################################
    
    def scriptToSpecialCharsRatio(self):
        v = self.specialCharacters()
        if self.pq is not None and v!=0:
            sscr = self.scriptLength()/v
        else: sscr = 0
        return sscr
    
    #######################################################################################
    #                       3.8 The ratio of total length of embedded                     #
    #                   scripts to total number of characters in HTML page                #
    #######################################################################################

    def scriptTobodyRatio(self):
        v = self.bodyLength()
        if self.pq is not None and v!=0:
            sbr = self.scriptLength()/v
        else: sbr = 0
        return sbr
    
    #######################################################################################
    #                       3.9 The ratio of total number of special                      #
    #                   characters to body length in URL's HTML page                      #
    #######################################################################################

    def bodyToSpecialCharRatio(self):
        v = self.bodyLength()
        if self.pq is not None and v!=0:
            bscr = self.specialCharacters()/v
        else: bscr = 0
        return bscr

    #######################################################################################
    #                               3.10 The page is online                               #
    #######################################################################################

    def urlIsLive(self):
        return self.response == 200
    
    #######################################################################################
    #  _____     __   ____ ___  __  __ ____ ___ _   _ _____                               #
    # |_ _\ \   / /  / ___/ _ \|  \/  | __ )_ _| \ | | ____|                              #
    #  | | \ \ / /  | |  | | | | |\/| |  _ \| ||  \| |  _|                                #
    #  | |  \ V /   | |__| |_| | |  | | |_) | || |\  | |___                               #
    # |___|_ \_(_)   \____\___/|_| _|_|____/___|_|_\_|_____|                              #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    def extract(self):
        data = {}
        data['File'] = "Unknown"
        data['bodyLength'] = self.bodyLength()
        data['bscr'] = self.bodyToSpecialCharRatio()
        data['dse'] = self.daysSinceExpiration()
        data['dsr'] = self.daysSinceRegistration()
        data['entropy'] = self.entropy()
        data['hasHttp'] = self.hasHttp()
        data['hasHttps'] = self.hasHttps()
        data['has_ip'] = self.has_ip()
        data['numDigits'] = self.numDigits()
        data['numImages'] = self.numImages()
        data['numLinks'] = self.numLinks()
        data['numParams'] = self.numParameters()
        data['numTitles'] = self.numTitles()
        data['num_%20'] = self.url.count("%20")
        data['num_@'] = self.url.count("@")
        data['sbr'] = self.scriptTobodyRatio()
        data['scriptLength'] = self.scriptLength()
        data['specialChars'] = self.specialCharacters()
        data['sscr'] = self.scriptToSpecialCharsRatio()
        data['urlIsLive'] = self.urlIsLive()
        data['urlLength'] = self.urlLength()
        
        return data