import os
import whois
import math
import re
import socket
import Levenshtein
import tldextract
import requests
import concurrent.futures
import dns.resolver
import time
import logging

from functools import wraps
from pyquery import PyQuery
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, urljoin
from urllib.request import urlopen
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def timer(func):
    """Record execution time of any functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        if args and hasattr(args[0], "exec_time"):
            args[0].exec_time += elapsed_time
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds, cumulative exec_time: {args[0].exec_time:.2f} seconds"
            )
        else:
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds (no instance with exec_time found)"
            )
        return result  # Return the original function's result

    return wrapper


except_funcs = [
    "get_state_and_page",
    "global_rank",
    "page_rank",
    "google_index",
    "dns_record",
    "count_internal_redirect",
    "count_external_redirect",
    "count_internal_error",
    "count_external_error",
]


def deadline(timeout):
    """Deadline execution time of any functions"""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    if func.__name__ == except_funcs[0]:  # get_state_and_page
                        return (0, None)
                    elif func.__name__ in except_funcs[1:3]:  # global_rank, page_rank
                        return -1
                    elif func.__name__ in except_funcs[3:5]:  # google_index, dns_record
                        return 1
                    else:  # count_internal_redirect, count_external_redirect, count_internal_error, count_external_error
                        return 0

        return wrapper

    return decorate


class URL_EXTRACTOR(object):
    # Cache
    global_rank_cache = {}
    page_rank_cache = {}
    whois_cache = {}
    dns_record_cache = {}

    @timer
    def __init__(self, url, label="Unknown", enable_logging=False):
        # Log and execution time informations
        self.exec_time = 0.0
        self.log_level = logging.INFO if enable_logging else logging.WARNING
        logging.getLogger().setLevel(self.log_level)

        # URL-Based Informations
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url
        self.url = url
        self.label = label
        self.p = urlparse(self.url)
        self.pq = None
        self.extracted = tldextract.extract(self.url)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.state, self.page = self.get_state_and_page()
        self.hostname = self.p.hostname or ""
        self.domain = self.extracted.domain + "." + self.extracted.suffix
        self.subdomain = self.extracted.subdomain
        self.tld = self.extracted.suffix
        self.path = self.p.path or ""
        self.query = self.p.query or ""
        self.scheme = self.p.scheme
        self.words_raw, self.words_raw_host, self.words_raw_path = (
            self.words_raw_extraction()
        )

        # Whois and PyQuery Informations
        self.res = self.get_whois()

        try:
            self.whois = whois.query(self.domain).__dict__
        except:
            self.whois = None

        # Page HTML Informations
        if self.page is not None and self.state == 1:
            try:
                self.soup = BeautifulSoup(self.page.content, "html.parser")
            except Exception as e:
                logging.error(f"BeautifulSoup parsing failed: {e}")
                self.soup = None
            try:
                self.pq = PyQuery(self.page.text)
            except:
                self.pq = None
        else:
            self.soup = None
            logging.info("Skipping BeautifulSoup parsing due to invalid page content")

        if self.soup:
            self.Text = (
                self.soup.get_text().encode("utf-8", errors="replace").decode("utf-8")
            )
        else:
            self.Text = ""
        self.Href = {"internals": [], "externals": [], "null": []}
        self.Link = {"internals": [], "externals": [], "null": []}
        self.Anchor = {"safe": [], "unsafe": [], "null": []}
        self.Media = {"internals": [], "externals": [], "null": []}
        self.Form = {"internals": [], "externals": [], "null": []}
        self.CSS = {"internals": [], "externals": [], "null": []}
        self.Favicon = {"internals": [], "externals": [], "null": []}
        self.IFrame = {"visible": [], "invisible": []}
        self.Title = ""
        self.Null_format = [
            "",
            "#",
            "#nothing",
            "#doesnotexist",
            "#null",
            "#void",
            "#whatever",
            "#content",
            "javascript::void(0)",
            "javascript::void(0);",
            "javascript::;",
            "javascript",
        ]

        try:
            self.get_hrefs_and_anchors()
        except KeyError:
            self.Href = {"internals": [], "externals": [], "null": []}
            self.Anchor = {"safe": [], "unsafe": [], "null": []}

        try:
            self.get_links()
            self.get_scripts()
        except KeyError:
            self.Link = {"internals": [], "externals": [], "null": []}

        try:
            self.get_media_imgs()
            self.get_media_embeds()
            self.get_media_audios()
            self.get_media_iframes()
        except KeyError:
            self.Media = {"internals": [], "externals": [], "null": []}

        try:
            self.get_css_links()
            self.get_css_styles()
        except KeyError:
            self.CSS = {"internals": [], "externals": [], "null": []}

        try:
            self.get_forms()
        except KeyError:
            self.Form = {"internals": [], "externals": [], "null": []}

        try:
            self.get_favicon()
        except KeyError:
            self.Favicon = {"internals": [], "externals": [], "null": []}

        try:
            self.get_iframes()
        except KeyError:
            self.IFrame = {"visible": [], "invisible": []}

        try:
            self.get_title()
        except KeyError:
            self.Title = ""

        # Other references
        self.allbrands_path = open(
            os.path.join(BASE_DIR, "scripts", "allbrands.txt"), "r"
        )
        self.allbrands = self.__txt_to_list()
        self.hints = [
            "wp",
            "login",
            "includes",
            "admin",
            "content",
            "site",
            "images",
            "js",
            "alibaba",
            "css",
            "myaccount",
            "dropbox",
            "themes",
            "plugins",
            "signin",
            "view",
        ]
        self.suspecious_tlds = [
            "fit",
            "tk",
            "gp",
            "ga",
            "work",
            "ml",
            "date",
            "wang",
            "men",
            "icu",
            "online",
            "click",  # Spamhaus
            "country",
            "stream",
            "download",
            "xin",
            "racing",
            "jetzt",
            "ren",
            "mom",
            "party",
            "review",
            "trade",
            "accountants",
            "science",
            "work",
            "ninja",
            "xyz",
            "faith",
            "zip",
            "cricket",
            "win",
            "accountant",
            "realtor",
            "top",
            "christmas",
            "gdn",  # Shady Top-Level Domains
            "link",  # Blue Coat Systems
            "asia",
            "club",
            "la",
            "ae",
            "exposed",
            "pe",
            "go.id",
            "rs",
            "k12.pa.us",
            "or.kr",
            "ce.ke",
            "audio",
            "gob.pe",
            "gov.az",
            "website",
            "bj",
            "mx",
            "media",
            "sa.gov.au",  # statistics
        ]
        self.OPR_API_key = "gk8cg0gckckwk8gso88ss4c888cs4csc480s00o8"

    @timer
    @deadline(5)
    def get_whois(self):
        if self.domain in URL_EXTRACTOR.whois_cache:
            return URL_EXTRACTOR.whois_cache[self.domain]
        try:
            result = whois.whois(self.domain)
            URL_EXTRACTOR.whois_cache[self.domain] = result
            return result
        except Exception as e:
            URL_EXTRACTOR.whois_cache[self.domain] = None
            return None

    @timer
    def __txt_to_list(self):
        list = []
        for line in self.allbrands_path:
            list.append(line.strip())
        self.allbrands_path.close()
        return list

    @timer
    def words_raw_extraction(self):
        w_domain = re.split(
            r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.domain or "").lower()
        )
        w_subdomain = re.split(
            r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.subdomain or "").lower()
        )
        w_path = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.path or "").lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None, raw_words))
        return raw_words, list(filter(None, w_host)), list(filter(None, w_path))

    @staticmethod
    @timer
    def is_valid_url(url):
        try:
            result = re.match(r"^https?://[^\s/$.?#].[^\s]*$", url)
            parsed = urlparse(url)
            return bool(result and parsed.scheme and parsed.netloc)
        except:
            return False

    @timer
    def normalize_url(self, link):
        if not link:
            return None
        # remove //
        link = (
            link.replace("//", "/", 1)
            if link.startswith(self.hostname + "//")
            else link
        )
        # add protocol if missing
        if not link.startswith(("http://", "https://")):
            link = urljoin(f"https://{self.hostname}", link)
        return link if self.is_valid_url(link) else None

    @timer
    def get_hrefs_and_anchors(self):
        """Extracts and categorizes all <a href=...> tags and anchors."""
        if self.soup:
            for href in self.soup.find_all("a", href=True):
                dots = [x.start(0) for x in re.finditer(r"\.", href["href"])]
                if (
                    self.hostname in href["href"]
                    or self.domain in href["href"]
                    or len(dots) == 1
                    or not href["href"].startswith("http")
                ):
                    if (
                        "#" in href["href"]
                        or "javascript" in href["href"].lower()
                        or "mailto" in href["href"].lower()
                    ):
                        self.Anchor["unsafe"].append(href["href"])
                    if not href["href"].startswith("http"):
                        if not href["href"].startswith("/"):
                            self.Href["internals"].append(
                                self.hostname + "/" + href["href"]
                            )
                        elif href["href"] in self.Null_format:
                            self.Href["null"].append(href["href"])
                        else:
                            self.Href["internals"].append(self.hostname + href["href"])
                else:
                    self.Href["externals"].append(href["href"])
                    self.Anchor["safe"].append(href["href"])

    @timer
    def get_links(self):
        """Extracts and categorizes all <link href=...> tags."""
        if self.soup:
            for link in self.soup.find_all("link", href=True):
                dots = [x.start(0) for x in re.finditer(r"\.", link["href"])]
                if (
                    self.hostname in link["href"]
                    or self.domain in link["href"]
                    or len(dots) == 1
                    or not link["href"].startswith("http")
                ):
                    if not link["href"].startswith("http"):
                        if not link["href"].startswith("/"):
                            self.Link["internals"].append(
                                self.hostname + "/" + link["href"]
                            )
                        elif link["href"] in self.Null_format:
                            self.Link["null"].append(link["href"])
                        else:
                            self.Link["internals"].append(self.hostname + link["href"])
                else:
                    self.Link["externals"].append(link["href"])

    @timer
    def get_scripts(self):
        """Extracts and categorizes all <script src=...> tags."""
        if self.soup:
            for script in self.soup.find_all("script", src=True):
                dots = [x.start(0) for x in re.finditer(r"\.", script["src"])]
                if (
                    self.hostname in script["src"]
                    or self.domain in script["src"]
                    or len(dots) == 1
                    or not script["src"].startswith("http")
                ):
                    if not script["src"].startswith("http"):
                        if not script["src"].startswith("/"):
                            self.Link["internals"].append(
                                self.hostname + "/" + script["src"]
                            )
                        elif script["src"] in self.Null_format:
                            self.Link["null"].append(script["src"])
                        else:
                            self.Link["internals"].append(self.hostname + script["src"])
                else:
                    self.Link["externals"].append(script["src"])

    @timer
    def get_media_imgs(self):
        """Extracts and categorizes all <img src=...> tags."""
        if self.soup:
            for img in self.soup.find_all("img", src=True):
                dots = [x.start(0) for x in re.finditer(r"\.", img["src"])]
                if (
                    self.hostname in img["src"]
                    or self.domain in img["src"]
                    or len(dots) == 1
                    or not img["src"].startswith("http")
                ):
                    if not img["src"].startswith("http"):
                        if not img["src"].startswith("/"):
                            self.Media["internals"].append(
                                self.hostname + "/" + img["src"]
                            )
                        elif img["src"] in self.Null_format:
                            self.Media["null"].append(img["src"])
                        else:
                            self.Media["internals"].append(self.hostname + img["src"])
                else:
                    self.Media["externals"].append(img["src"])

    @timer
    def get_media_audios(self):
        """Extracts and categorizes all <audio src=...> tags."""
        if self.soup:
            for audio in self.soup.find_all("audio", src=True):
                dots = [x.start(0) for x in re.finditer(r"\.", audio["src"])]
                if (
                    self.hostname in audio["src"]
                    or self.domain in audio["src"]
                    or len(dots) == 1
                    or not audio["src"].startswith("http")
                ):
                    if not audio["src"].startswith("http"):
                        if not audio["src"].startswith("/"):
                            self.Media["internals"].append(
                                self.hostname + "/" + audio["src"]
                            )
                        elif audio["src"] in self.Null_format:
                            self.Media["null"].append(audio["src"])
                        else:
                            self.Media["internals"].append(self.hostname + audio["src"])
                else:
                    self.Media["externals"].append(audio["src"])

    @timer
    def get_media_embeds(self):
        """Extracts and categorizes all <embed src=...> tags."""
        if self.soup:
            for embed in self.soup.find_all("embed", src=True):
                dots = [x.start(0) for x in re.finditer(r"\.", embed["src"])]
                if (
                    self.hostname in embed["src"]
                    or self.domain in embed["src"]
                    or len(dots) == 1
                    or not embed["src"].startswith("http")
                ):
                    if not embed["src"].startswith("http"):
                        if not embed["src"].startswith("/"):
                            self.Media["internals"].append(
                                self.hostname + "/" + embed["src"]
                            )
                        elif embed["src"] in self.Null_format:
                            self.Media["null"].append(embed["src"])
                        else:
                            self.Media["internals"].append(self.hostname + embed["src"])
                else:
                    self.Media["externals"].append(embed["src"])

    @timer
    def get_media_iframes(self):
        """Extracts and categorizes all <iframe src=...> tags."""
        if self.soup:
            for i_frame in self.soup.find_all("iframe", src=True):
                dots = [x.start(0) for x in re.finditer(r"\.", i_frame["src"])]
                if (
                    self.hostname in i_frame["src"]
                    or self.domain in i_frame["src"]
                    or len(dots) == 1
                    or not i_frame["src"].startswith("http")
                ):
                    if not i_frame["src"].startswith("http"):
                        if not i_frame["src"].startswith("/"):
                            self.Media["internals"].append(
                                self.hostname + "/" + i_frame["src"]
                            )
                        elif i_frame["src"] in self.Null_format:
                            self.Media["null"].append(i_frame["src"])
                        else:
                            self.Media["internals"].append(
                                self.hostname + i_frame["src"]
                            )
                else:
                    self.Media["externals"].append(i_frame["src"])

    @timer
    def get_forms(self):
        """Extracts and categorizes all <form action=...> tags."""
        if self.soup:
            for form in self.soup.find_all("form", action=True):
                dots = [x.start(0) for x in re.finditer(r"\.", form["action"])]
                if (
                    self.hostname in form["action"]
                    or self.domain in form["action"]
                    or len(dots) == 1
                    or not form["action"].startswith("http")
                ):
                    if not form["action"].startswith("http"):
                        if not form["action"].startswith("/"):
                            self.Form["internals"].append(
                                self.hostname + "/" + form["action"]
                            )
                        elif (
                            form["action"] in self.Null_format
                            or form["action"] == "about:blank"
                        ):
                            self.Form["null"].append(form["action"])
                        else:
                            self.Form["internals"].append(
                                self.hostname + form["action"]
                            )
                else:
                    self.Form["externals"].append(form["action"])

    @timer
    def get_css_links(self):
        """Extracts and categorizes all <link rel='stylesheet'> tags."""
        if self.soup:
            for link in self.soup.find_all("link", rel="stylesheet"):
                dots = [x.start(0) for x in re.finditer(r"\.", link["href"])]
                if (
                    self.hostname in link["href"]
                    or self.domain in link["href"]
                    or len(dots) == 1
                    or not link["href"].startswith("http")
                ):
                    if not link["href"].startswith("http"):
                        if not link["href"].startswith("/"):
                            self.CSS["internals"].append(
                                self.hostname + "/" + link["href"]
                            )
                        elif link["href"] in self.Null_format:
                            self.CSS["null"].append(link["href"])
                        else:
                            self.CSS["internals"].append(self.hostname + link["href"])
                else:
                    self.CSS["externals"].append(link["href"])

    @timer
    def get_css_styles(self):
        """Extracts and categorizes all <style> tags."""
        if self.soup:
            for style in self.soup.find_all("style", type="text/css"):
                try:
                    start = str(style[0]).index("@import url(")
                    end = str(style[0]).index(")")
                    css = str(style[0])[start + 12 : end]
                    dots = [x.start(0) for x in re.finditer(r"\.", css)]
                    if (
                        self.hostname in css
                        or self.domain in css
                        or len(dots) == 1
                        or not css.startswith("http")
                    ):
                        if not css.startswith("http"):
                            if not css.startswith("/"):
                                self.CSS["internals"].append(self.hostname + "/" + css)
                            elif css in self.Null_format:
                                self.CSS["null"].append(css)
                            else:
                                self.CSS["internals"].append(self.hostname + css)
                    else:
                        self.CSS["externals"].append(css)
                except:
                    continue

    @timer
    def get_favicon(self):
        """Extracts and categorizes favicon links."""
        if self.soup:
            for head in self.soup.find_all("head"):
                for head_link in self.soup.find_all("link", href=True):
                    dots = [x.start(0) for x in re.finditer(r"\.", head_link["href"])]
                    if (
                        self.hostname in head_link["href"]
                        or len(dots) == 1
                        or self.domain in head_link["href"]
                        or not head_link["href"].startswith("http")
                    ):
                        if not head_link["href"].startswith("http"):
                            if not head_link["href"].startswith("/"):
                                self.Favicon["internals"].append(
                                    self.hostname + "/" + head_link["href"]
                                )
                            elif head_link["href"] in self.Null_format:
                                self.Favicon["null"].append(head_link["href"])
                            else:
                                self.Favicon["internals"].append(
                                    self.hostname + head_link["href"]
                                )
                    else:
                        self.Favicon["externals"].append(head_link["href"])

                for head_link in self.soup.find_all(
                    "link", {"href": True, "rel": True}
                ):
                    isicon = False
                    if isinstance(head_link["rel"], list):
                        for e_rel in head_link["rel"]:
                            if e_rel.endswith("icon"):
                                isicon = True
                    else:
                        if head_link["rel"].endswith("icon"):
                            isicon = True
                    if isicon:
                        dots = [
                            x.start(0) for x in re.finditer(r"\.", head_link["href"])
                        ]
                        if (
                            self.hostname in head_link["href"]
                            or len(dots) == 1
                            or self.domain in head_link["href"]
                            or not head_link["href"].startswith("http")
                        ):
                            if not head_link["href"].startswith("http"):
                                if not head_link["href"].startswith("/"):
                                    self.Favicon["internals"].append(
                                        self.hostname + "/" + head_link["href"]
                                    )
                                elif head_link["href"] in self.Null_format:
                                    self.Favicon["null"].append(head_link["href"])
                                else:
                                    self.Favicon["internals"].append(
                                        self.hostname + head_link["href"]
                                    )
                        else:
                            self.Favicon["externals"].append(head_link["href"])

    @timer
    def get_iframes(self):
        """Extracts and categorizes iframes as visible/invisible based on width/height."""
        if self.soup:
            for i_frame in self.soup.find_all(
                "iframe", width=True, height=True, frameborder=True
            ):
                if (
                    i_frame["width"] == "0"
                    and i_frame["height"] == "0"
                    and i_frame["frameborder"] == "0"
                ):
                    self.IFrame["invisible"].append(i_frame)
                else:
                    self.IFrame["visible"].append(i_frame)
            for i_frame in self.soup.find_all(
                "iframe", width=True, height=True, border=True
            ):
                if (
                    i_frame["width"] == "0"
                    and i_frame["height"] == "0"
                    and i_frame["border"] == "0"
                ):
                    self.IFrame["invisible"].append(i_frame)
                else:
                    self.IFrame["visible"].append(i_frame)
            for i_frame in self.soup.find_all(
                "iframe", width=True, height=True, style=True
            ):
                if (
                    i_frame["width"] == "0"
                    and i_frame["height"] == "0"
                    and i_frame["style"] == "border:none;"
                ):
                    self.IFrame["invisible"].append(i_frame)
                else:
                    self.IFrame["visible"].append(i_frame)

    @timer
    def get_title(self):
        """Returns the page title."""
        if self.soup:
            try:
                self.Title = self.soup.title.string
            except:
                self.Title = ""

    #######################################################################################
    #  ___     _   _ ____  _     _                                                        #
    # |_ _|   | | | |  _ \| |   ( )___                                                    #
    #  | |    | | | | |_) | |   |// __|                                                   #
    #  | | _  | |_| |  _ <| |___  \__ \                                                   #
    # |___(_)__\___/|_| \_\_____| |___/__  _____ ____                                     #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                            1.1 Entropy of URL                                       #
    #######################################################################################

    @timer
    def entropy(self):
        str = self.url.strip()
        prob = [float(str.count(c)) / len(str) for c in dict.fromkeys(list(str))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    #######################################################################################
    #                           1.2 Having IP address in hostname                         #
    #######################################################################################

    @timer
    def having_ip_address(self):
        match = re.search(
            r"(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\."
            r"([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|"  # IPv4
            r"((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|"  # IPv4 in hexadecimal
            r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|"
            r"[0-9a-fA-F]{7}",
            self.url,
        )  # Ipv6
        if match:
            return 1
        else:
            return 0

    #######################################################################################
    #                     1.3 Total number of digits in URL string                        #
    #######################################################################################

    @timer
    def count_digits(self):
        return len(re.sub(r"[^0-9]", "", self.url))

    #######################################################################################
    #                    1.4 Total number of characters in URL string                     #
    #######################################################################################

    @timer
    def url_len(self):
        return len(self.url)

    @timer
    def hostname_len(self):
        return len(self.hostname)

    #######################################################################################
    #                    1.5 Total number of query parameters in URL                      #
    #######################################################################################

    @timer
    def count_parameters(self):
        params = self.url.split("&")
        return len(params) - 1

    #######################################################################################
    #                         1.6 Total Number of Fragments in URL                        #
    #######################################################################################

    @timer
    def count_fragments(self):
        fragments = self.url.split("#")
        return len(fragments) - 1

    #######################################################################################
    #                         1.7 URL shortening                                          #
    #######################################################################################

    @timer
    def has_shortening_service(self):
        match = re.search(
            r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
            r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
            r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
            r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
            r"db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
            r"q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
            r"x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
            r"tr\.im|link\.zip\.net",
            self.url,
        )
        if match:
            return 1
        else:
            return 0

    #######################################################################################
    #                         1.8 Count at (@) symbol at base url                        #
    #######################################################################################

    @timer
    def count_at(self):
        return self.url.count("@")

    #######################################################################################
    #                         1.9 Count comma (,) symbol at base url                    #
    #######################################################################################

    @timer
    def count_comma(self):
        return self.url.count(",")

    #######################################################################################
    #                         1.10 Count dollar ($) symbol at base url                    #
    #######################################################################################

    @timer
    def count_dollar(self):
        return self.url.count("$")

    #######################################################################################
    #                         1.11 Count semicolumn (;) symbol at base url                #
    #######################################################################################

    @timer
    def count_semicolumn(self):
        return self.url.count(";")

    #######################################################################################
    #                         1.12 Count (space, %20) symbol at base url                  #
    #######################################################################################

    @timer
    def count_space(self):
        return self.url.count(" ") + self.url.count("%20")

    #######################################################################################
    #                         1.13 Count and (&) symbol at base url                       #
    #######################################################################################

    @timer
    def count_and(self):
        return self.url.count("&")

    #######################################################################################
    #                         1.14 Count redirection (//) symbol at full url              #
    #######################################################################################

    @timer
    def count_double_slash(self):
        positions = [x.start(0) for x in re.finditer("//", self.url)]
        if positions and positions[-1] > 6:
            return 1
        else:
            return 0

    #######################################################################################
    #                         1.15 Count slash (/) symbol at full url                     #
    #######################################################################################

    @timer
    def count_slash(self):
        return self.url.count("/")

    #######################################################################################
    #                         1.16 Count equal (=) symbol at base url                     #
    #######################################################################################

    @timer
    def count_equal(self):
        return self.url.count("=")

    #######################################################################################
    #                         1.17 Count percentage (%) symbol at base url                #
    #######################################################################################

    @timer
    def count_percentage(self):
        return self.url.count("%")

    #######################################################################################
    #                         1.18 Count exclamation (?) symbol at base url               #
    #######################################################################################

    @timer
    def count_exclamation(self):
        return self.url.count("?")

    #######################################################################################
    #                         1.19 Count underscore (_) symbol at base url                #
    #######################################################################################

    @timer
    def count_underscore(self):
        return self.url.count("_")

    #######################################################################################
    #                         1.20 Count hyphens (-) symbol at base url                   #
    #######################################################################################

    @timer
    def count_hyphens(self):
        return self.url.count("-")

    #######################################################################################
    #                         1.21 Count number of dots in hostname                       #
    #######################################################################################

    @timer
    def count_dots(self):
        return self.url.count(".")

    #######################################################################################
    #                         1.22 Count number of colon (:) symbol at base url           #
    #######################################################################################

    @timer
    def count_colon(self):
        return self.url.count(":")

    #######################################################################################
    #                         1.23 Count number of stars (*) symbol at base url           #
    #######################################################################################

    @timer
    def count_star(self):
        return self.url.count("*")

    #######################################################################################
    #                         1.24 Count number of OR (|) symbol at base url              #
    #######################################################################################

    @timer
    def count_or(self):
        return self.url.count("|")

    #######################################################################################
    #                         1.25 Path entension != .txt/.exe                            #
    #######################################################################################

    @timer
    def has_path_txt_extension(self):
        if self.path.endswith(".txt"):
            return 1
        return 0

    @timer
    def has_path_exe_extension(self):
        if self.path.endswith(".exe"):
            return 1
        return 0

    #######################################################################################
    #                         1.26 Count number of http or https in url path              #
    #######################################################################################

    @timer
    def count_http_token(self):
        combined = self.path + (("?" + self.query) if self.query else "")
        count = len(re.findall(r"https?", combined))
        return count

    #######################################################################################
    #                         1.27 Uses https protocol                                    #
    #######################################################################################

    @timer
    def has_https(self):
        if self.scheme == "https":
            return 0
        return 1

    #######################################################################################
    #                   1.28 Checks if tilde symbol exist in webpage URL                  #
    #######################################################################################

    @timer
    def count_tilde(self):
        if self.url.count("~") > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.29 Number of phish-hints in url path                            #
    #######################################################################################

    @timer
    def count_phish_hints(self):
        count = 0
        for hint in self.hints:
            count += self.path.lower().count(hint)
        return count

    #######################################################################################
    #                   1.30 Check if TLD exists in the path                              #
    #######################################################################################

    @timer
    def has_tld_in_path(self):
        if self.path.lower().count(self.tld) > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.31 Check if TLD exists in the path                              #
    #######################################################################################

    @timer
    def has_tld_in_subdomain(self):
        if self.subdomain.count(self.tld) > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.32 Check if TLD in bad position                                 #
    #######################################################################################

    @timer
    def tld_in_bad_position(self):
        if (
            self.tld_in_path(self.tld, self.path) == 1
            or self.tld_in_subdomain(self.tld, self.subdomain) == 1
        ):
            return 1
        return 0

    #######################################################################################
    #                  1.33 Abnormal subdomain starting with wwww-, wwNN                  #
    #######################################################################################

    @timer
    def has_abnormal_subdomain(self):
        if re.search(r"(http[s]?://(w[w]?|\d))([w]?(\d|-))", self.url):
            return 1
        return 0

    #######################################################################################
    #                           1.34 Number of redirection                                #
    #######################################################################################

    @timer
    def count_redirection(self):
        try:
            return len(self.page.history)
        except AttributeError:
            return -1

    #######################################################################################
    #                   1.35 Number of redirection to different domains                   #
    #######################################################################################

    @timer
    def count_external_redirection(self):
        count = 0
        try:
            if len(self.page.history) == 0:
                return 0
            else:
                for i, response in enumerate(self.page.history, 1):
                    if self.domain.lower() not in response.url.lower():
                        count += 1
                    return count
        except AttributeError:
            return -1

    #######################################################################################
    #                           1.36 Consecutive Character Repeat                         #
    #######################################################################################

    @timer
    def char_repeat(self):
        def __all_same(items):
            return all(x == items[0] for x in items)

        repeat = {"2": 0, "3": 0, "4": 0, "5": 0}
        part = [2, 3, 4, 5]

        for word in self.words_raw:
            for char_repeat_count in part:
                for i in range(len(word) - char_repeat_count + 1):
                    sub_word = word[i : i + char_repeat_count]
                    if __all_same(sub_word):
                        repeat[str(char_repeat_count)] = (
                            repeat[str(char_repeat_count)] + 1
                        )
        return sum(list(repeat.values()))

    #######################################################################################
    #                              1.37 Puny code in domain                               #
    #######################################################################################

    @timer
    def has_punycode(self):
        if self.url.startswith("http://xn--") or self.url.startswith("http://xn--"):
            return 1
        else:
            return 0

    #######################################################################################
    #                              1.38 Domain in brand list                              #
    #######################################################################################

    @timer
    def has_domain_in_brand(self):
        if self.words_raw_host[0] in self.allbrands:
            return 1
        else:
            return 0

    @timer
    def has_domain_in_brand1(self):
        for d in self.allbrands:
            if len(Levenshtein.editops(self.words_raw_host[0].lower(), d.lower())) < 2:
                return 1
        return 0

    #######################################################################################
    #                              1.39 Brand name in path/domain                         #
    #######################################################################################

    @timer
    def has_brand_in_path(self):
        for b in self.allbrands:
            if "." + b + "." in self.path and b not in self.domain:
                return 1
        return 0

    @timer
    def has_brand_in_subdomain(self):
        subdomain_components = self.subdomain.split(".") if self.subdomain else []
        for b in self.allbrands:
            if b in subdomain_components and b not in self.domain:
                return 1
        return 0

    #######################################################################################
    #                              1.40 Count www in url words                            #
    #######################################################################################

    @timer
    def count_www(self):
        count = 0
        for word in self.words_raw:
            if not word.find("www") == -1:
                count += 1
        return count

    #######################################################################################
    #                              1.41 Count com in url words                            #
    #######################################################################################

    @timer
    def count_com(self):
        count = 0
        for word in self.words_raw:
            if not word.find("com") == -1:
                count += 1
        return count

    #######################################################################################
    #                          1.42 Check port presence in domain                         #
    #######################################################################################

    @timer
    def has_port(self):
        if re.search(
            r"^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",
            self.url,
        ):
            return 1
        return 0

    #######################################################################################
    #                             1.43 Length of raw word list                            #
    #######################################################################################

    @timer
    def length_word_raw(self):
        return len(self.words_raw)

    #######################################################################################
    #                   1.44 Count average word length in raw word list                   #
    #######################################################################################

    @timer
    def average_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return sum(len(word) for word in self.words_raw) / len(self.words_raw)

    @timer
    def average_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return sum(len(word) for word in self.words_raw_host) / len(self.words_raw_host)

    @timer
    def average_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return sum(len(word) for word in self.words_raw_path) / len(self.words_raw_path)

    #######################################################################################
    #                   1.45 longest word length in raw word list                         #
    #######################################################################################

    @timer
    def longest_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return max(len(word) for word in self.words_raw)

    @timer
    def longest_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return max(len(word) for word in self.words_raw_host)

    @timer
    def longest_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return max(len(word) for word in self.words_raw_path)

    #######################################################################################
    #                   1.46 Shortest word length in word list                            #
    #######################################################################################

    @timer
    def shortest_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return min(len(word) for word in self.words_raw)

    @timer
    def shortest_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return min(len(word) for word in self.words_raw_host)

    @timer
    def shortest_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return min(len(word) for word in self.words_raw_path)

    #######################################################################################
    #                              1.47 Prefix suffix                                     #
    #######################################################################################

    @timer
    def has_prefix_suffix(self):
        if re.findall(r"https?://[^\-]+-[^\-]+/", self.url):
            return 1
        else:
            return 0

    #######################################################################################
    #                              1.48 Count subdomain                                   #
    #######################################################################################

    @timer
    def count_subdomain(self):
        if len(re.findall(r"\.", self.url)) == 1:
            return 1
        elif len(re.findall(r"\.", self.url)) == 2:
            return 2
        else:
            return 3

    #######################################################################################
    #                             1.49 Statistical report                                 #
    #######################################################################################

    @timer
    def has_statistical_report(self):
        url_match = re.search(
            r"at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly",
            self.url,
        )
        try:
            ip_address = socket.gethostbyname(self.domain)
            ip_match = re.search(
                r"146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|"
                r"107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|"
                r"118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|"
                r"216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|"
                r"34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|"
                r"216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42",
                ip_address,
            )
            if url_match or ip_match:
                return 1
            elif ip_address == "125.235.4.59":
                return 2
            else:
                return 0
        except socket.gaierror:
            return 2
        except Exception:
            return 2

    #######################################################################################
    #                               1.50 Suspecious TLD                                   #
    #######################################################################################

    @timer
    def has_suspecious_tld(self):
        if self.tld in self.suspecious_tlds:
            return 1
        return 0

    #######################################################################################
    #                        1.51 Ratio of digits in url                                  #
    #######################################################################################

    @timer
    def ratio_digits_url(self):
        return len(re.sub(r"[^0-9]", "", self.url)) / len(self.url)

    #######################################################################################
    #                        1.52 Ratio of digits in hostname                             #
    #######################################################################################

    @timer
    def ratio_digits_hostname(self):
        if not self.hostname:
            return 0
        return len(re.sub(r"[^0-9]", "", self.hostname)) / len(self.hostname)

    #######################################################################################
    #  ___ ___      ____ ___  _   _ _____ _____ _   _ _____ _                             #
    # |_ _|_ _|    / ___/ _ \| \ | |_   _| ____| \ | |_   _( )___                         #
    #  | | | |    | |  | | | |  \| | | | |  _| |  \| | | | |// __|                        #
    #  | | | | _  | |__| |_| | |\  | | | | |___| |\  | | |   \__ \                        #
    # |___|___(_)_ \____\___/|_| \_| |_|_|_____|_|_\_| |_|   |___/                        #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                 2.1 Number of hyperlinks present in a website                       #
    #######################################################################################

    @timer
    def count_hyperlinks(self):
        return (
            len(self.Href["internals"])
            + len(self.Href["externals"])
            + len(self.Link["internals"])
            + len(self.Link["externals"])
            + len(self.Media["internals"])
            + len(self.Media["externals"])
            + len(self.Form["internals"])
            + len(self.Form["externals"])
            + len(self.CSS["internals"])
            + len(self.CSS["externals"])
            + len(self.Favicon["internals"])
            + len(self.Favicon["externals"])
        )

    #######################################################################################
    #                           2.2 Internal hyperlinks ratio                             #
    #######################################################################################

    @timer
    def count_internal_hyperlinks(self):
        return (
            len(self.Href["internals"])
            + len(self.Link["internals"])
            + len(self.Media["internals"])
            + len(self.Form["internals"])
            + len(self.CSS["internals"])
            + len(self.Favicon["internals"])
        )

    @timer
    def ratio_internal_hyperlinks(self):
        total = self.count_hyperlinks()
        if total == 0:
            return 0
        else:
            return self.count_internal_hyperlinks() / total

    #######################################################################################
    #                           2.3 External hyperlinks ratio                             #
    #######################################################################################

    @timer
    def count_external_hyperlinks(self):
        return (
            len(self.Href["externals"])
            + len(self.Link["externals"])
            + len(self.Media["externals"])
            + len(self.Form["externals"])
            + len(self.CSS["externals"])
            + len(self.Favicon["externals"])
        )

    @timer
    def ratio_external_hyperlinks(self):
        total = self.count_hyperlinks()
        if total == 0:
            return 0
        else:
            self.count_external_hyperlinks() / total

    #######################################################################################
    #                           2.4 Null hyperlinks ratio                                 #
    #######################################################################################

    @timer
    def count_null_hyperlinks(self):
        return (
            len(self.Href["null"])
            + len(self.Link["null"])
            + len(self.Media["null"])
            + len(self.Form["null"])
            + len(self.CSS["null"])
            + len(self.Favicon["null"])
        )

    @timer
    def ratio_null_hyperlinks(self):
        total = self.count_hyperlinks()
        if total == 0:
            return 0
        else:
            self.nb_null_hyperlinks() / total

    #######################################################################################
    #                               2.5 External CSS                                      #
    #######################################################################################

    @timer
    def count_external_css(self):
        return len(self.CSS["externals"])

    #######################################################################################
    #                           2.6 Internal redirections ratio                           #
    #######################################################################################

    @timer
    @deadline(10)
    def count_internal_redirect(self):
        count = 0
        for link in self.Href["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Link["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Media["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Form["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.CSS["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Favicon["internals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        return count

    @timer
    def ratio_internal_redirection(self):
        internals = self.nb_internal_hyperlinks()
        if internals > 0:
            return self.count_internal_redirect() / internals
        else:
            return 0

    #######################################################################################
    #                           2.7 External redirections ratio                           #
    #######################################################################################

    @timer
    @deadline(10)
    def count_external_redirect(self):
        count = 0
        for link in self.Href["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Link["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Media["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Form["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.CSS["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        for link in self.Favicon["externals"]:
            try:
                link = self.normalize_url(link)
                r = requests.get(link, headers=self.headers, timeout=5)
                if len(r.history) > 0:
                    count += 1
            except:
                continue
        return count

    @timer
    def ratio_external_redirection(self):
        externals = self.count_external_hyperlinks()
        if externals > 0:
            return self.count_external_redirect() / externals
        else:
            return 0

    #######################################################################################
    #                           2.8 Generates internal errors                             #
    #######################################################################################

    @timer
    @deadline(10)
    def count_internal_error(self):
        count = 0
        for link in self.Href["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Link["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Media["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Form["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.CSS["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Favicon["internals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        return count

    @timer
    def ratio_internal_errors(self):
        internals = self.nb_internal_hyperlinks()
        if internals > 0:
            return self.count_internal_error() / internals
        return 0

    #######################################################################################
    #                           2.9 Generates external errors                             #
    #######################################################################################

    @timer
    @deadline(10)
    def count_external_error(self):
        count = 0
        for link in self.Href["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Link["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Media["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Form["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.CSS["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        for link in self.Favicon["externals"]:
            try:
                link = self.normalize_url(link)
                if (
                    requests.get(link, headers=self.headers, timeout=5).status_code
                    >= 400
                ):
                    count += 1
            except:
                continue
        return count

    @timer
    def ratio_external_errors(self):
        externals = self.count_external_hyperlinks()
        if externals > 0:
            return self.count_external_error() / externals
        return 0

    #######################################################################################
    #                           2.10 Having login form link                                #
    #######################################################################################

    @timer
    def has_login_form(self):
        p = re.compile(r"([a-zA-Z0-9\_])+.php")
        if len(self.Form["externals"]) > 0 or len(self.Form["null"]) > 0:
            return 1
        for form in self.Form["internals"] + self.Form["externals"]:
            if p.match(form) != None:
                return 1
        return 0

    #######################################################################################
    #                           2.11 Having external favicon                              #
    #######################################################################################

    @timer
    def has_external_favicon(self):
        if len(self.Favicon["externals"]) > 0:
            return 1
        return 0

    #######################################################################################
    #                               2.12 Submitting to email                              #
    #######################################################################################

    @timer
    def has_submitting_to_email(self):
        for form in self.Form["internals"] + self.Form["externals"]:
            if "mailto:" in form or "mail()" in form:
                return 1
            else:
                return 0
        return 0

    #######################################################################################
    #                           2.13 Percentile of internal media                         #
    #######################################################################################

    @timer
    def percentile_internal_media(self):
        total = len(self.Media["internals"]) + len(self.Media["externals"])
        internals = len(self.Media["internals"])
        try:
            percentile = internals / float(total) * 100
        except:
            return 0

        return percentile

    #######################################################################################
    #                           2.14 Percentile of external media                         #
    #######################################################################################

    @timer
    def percentile_external_media(self):
        total = len(self.Media["internals"]) + len(self.Media["externals"])
        externals = len(self.Media["externals"])
        try:
            percentile = externals / float(total) * 100
        except:
            return 0

        return percentile

    #######################################################################################
    #                               2.15 Check for empty title                            #
    #######################################################################################

    @timer
    def empty_title(self):
        if self.Title:
            return 0
        return 1

    #######################################################################################
    #                              2.16 Percentile of safe anchor                         #
    #######################################################################################

    @timer
    def percentile_safe_anchor(self):
        total = len(self.Anchor["safe"]) + len(self.Anchor["unsafe"])
        unsafe = len(self.Anchor["unsafe"])
        try:
            percentile = unsafe / float(total) * 100
        except:
            return 0
        return percentile

    #######################################################################################
    #                           2.17 Percentile of internal links                         #
    #######################################################################################

    @timer
    def percentile_internal_links(self):
        total = len(self.Link["internals"]) + len(self.Link["externals"])
        internals = len(self.Link["internals"])
        try:
            percentile = internals / float(total) * 100
        except:
            return 0
        return percentile

    #######################################################################################
    #                               2.18 Server form handler                              #
    #######################################################################################

    @timer
    def has_srv_form_handler(self):
        if len(self.Form["null"]) > 0:
            return 1
        return 0

    #######################################################################################
    #                               2.19 IFrame redirection                               #
    #######################################################################################

    @timer
    def has_iframe(self):
        if len(self.IFrame["invisible"]) > 0:
            return 1
        return 0

    #######################################################################################
    #                               2.20 On mouse action                                  #
    #######################################################################################

    @timer
    def has_on_mouse_action(self):
        if 'onmouseover="window.status=' in str(self.Text).lower().replace(" ", ""):
            return 1
        else:
            return 0

    #######################################################################################
    #                               2.21 Pop up window                                    #
    #######################################################################################

    @timer
    def has_popup_window(self):
        if "prompt(" in str(self.Text).lower():
            return 1
        else:
            return 0

    #######################################################################################
    #                               2.22 Right_click action                               #
    #######################################################################################

    @timer
    def has_right_click(self):
        if re.findall(r"event.button ?== ?2", self.Text):
            return 1
        else:
            return 0

    #######################################################################################
    #                               2.23 Domain in page title                             #
    #######################################################################################

    @timer
    def has_domain_in_title(self):
        if not self.domain or not self.Title:
            return 1
        if self.domain.lower() in self.Title.lower():
            return 0
        return 1

    #######################################################################################
    #                           2.24 Domain after copyright logo                          #
    #######################################################################################

    @timer
    def has_domain_with_copyright(self):
        try:
            m = re.search(
                "(\N{COPYRIGHT SIGN}|\N{TRADE MARK SIGN}|\N{REGISTERED SIGN})",
                self.Text,
            )
            _copyright = self.Text[m.span()[0] - 50 : m.span()[0] + 50]
            if self.domain.lower() in _copyright.lower():
                return 0
            else:
                return 1
        except:
            return 0

    #######################################################################################
    #                  2.25 Total number of characters in URL's HTML page                 #
    #######################################################################################

    @timer
    def body_length(self):
        if self.pq is not None:
            return len(self.pq("html").text()) if self.state else 0
        else:
            return 0

    #######################################################################################
    #                 2.26 Total number of HI-H6 titles in URL's HTML page                #
    #######################################################################################

    @timer
    def nb_titles(self):
        if self.pq is not None:
            titles = ["h{}".format(i) for i in range(7)]
            titles = [self.pq(i).items() for i in titles]
            return len([item for s in titles for item in s])
        else:
            return 0

    #######################################################################################
    #              2.27 Total number of images embedded in URL's HTML page                #
    #######################################################################################

    @timer
    def nb_images(self):
        if self.pq is not None:
            return len([i for i in self.pq("img").items()])
        else:
            return 0

    #######################################################################################
    #              2.28 Total number of links embedded in URL's HTML page                 #
    #######################################################################################

    @timer
    def nb_links(self):
        if self.pq is not None:
            return len([i for i in self.pq("a").items()])
        else:
            return 0

    #######################################################################################
    #       2.29 Total number of characters in embedded scripts in URL's HTML page        #
    #######################################################################################

    @timer
    def script_length(self):
        if self.pq is not None:
            return len(self.pq("script").text())
        else:
            return 0

    #######################################################################################
    #              2.30 Total number of special characters in URL's HTML page             #
    #######################################################################################

    @timer
    def count_special_characters(self):
        if self.pq is not None:
            bodyText = self.pq("html").text()
            schars = [i for i in bodyText if not i.isdigit() and not i.isalpha()]
            return len(schars)
        else:
            return 0

    #######################################################################################
    #                      2.31 The ratio of total length of embedded                     #
    #                       scripts to special characters in HTML page                    #
    #######################################################################################

    @timer
    def ratio_script_to_special_chars(self):
        v = self.count_special_characters()
        if self.pq is not None and v != 0:
            sscr = self.script_length() / v
        else:
            sscr = 0
        return sscr

    #######################################################################################
    #                      2.32 The ratio of total length of embedded                     #
    #                   scripts to total number of characters in HTML page                #
    #######################################################################################

    @timer
    def ratio_script_to_body(self):
        v = self.body_length()
        if self.pq is not None and v != 0:
            sbr = self.script_length() / v
        else:
            sbr = 0
        return sbr

    #######################################################################################
    #                      2.33 The ratio of total number of special                      #
    #                   characters to body length in URL's HTML page                      #
    #######################################################################################

    @timer
    def ratio_body_to_special_char(self):
        v = self.body_length()
        if self.pq is not None and v != 0:
            bscr = self.count_special_characters() / v
        else:
            bscr = 0
        return bscr

    #######################################################################################
    #                              2.34 The page is alive                                 #
    #######################################################################################

    @timer
    @deadline(5)
    def get_state_and_page(self):
        page = None
        try:
            page = requests.get(
                self.url, headers=self.headers, timeout=5, allow_redirects=False
            )
        except:
            logging.info(f"Fail to fetch: {self.url}")
        if page and page.status_code == 200 and page.content not in [b"", b" "]:
            return 1, page
        return 0, page

    #######################################################################################
    #  ___ ___ ___     _______  _______ _____ ____  _   _    _    _                       #
    # |_ _|_ _|_ _|   | ____\ \/ /_   _| ____|  _ \| \ | |  / \  | |                      #
    #  | | | | | |    |  _|  \  /  | | |  _| | |_) |  \| | / _ \ | |                      #
    #  | | | | | | _  | |___ /  \  | | | |___|  _ <| |\  |/ ___ \| |___                   #
    # |___|___|___(_) |_____/_/\_\ |_|_|_____|_| \_\_| \_/_/   \_\_____|                  #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                             3.1 Datetime list converter                             #
    #######################################################################################

    @staticmethod
    @timer
    def normalize_datetime_list(date_list):
        """Convert a list of mixed naive and aware datetimes to a list of aware datetimes in UTC."""
        if not date_list:
            return []

        normalized_dates = []
        for dt in date_list:
            if dt is None:
                continue
            # Convert naive datetimes to aware ones (UTC)
            if dt.tzinfo is None:
                normalized_dates.append(dt.replace(tzinfo=timezone.utc))
            else:
                normalized_dates.append(dt.astimezone(timezone.utc))

        return normalized_dates

    #######################################################################################
    #                             3.2 Domain registration age                             #
    #######################################################################################

    @timer
    @deadline(5)
    def domain_registration_length(self):
        try:
            expiration_date = self.res.expiration_date

            # Handle case where expiration_date is a list
            if isinstance(expiration_date, list):
                # Normalize all dates to make them timezone-aware with UTC
                normalized_dates = self.normalize_datetime_list(expiration_date)
                if not normalized_dates:
                    return 0  # No valid data
                expiration_date = min(normalized_dates)
            elif expiration_date is None:
                return 0  # No data
            elif expiration_date.tzinfo is None:
                # Single naive datetime
                expiration_date = expiration_date.replace(tzinfo=timezone.utc)
            else:
                # Single aware datetime but potentially in a different timezone
                expiration_date = expiration_date.astimezone(timezone.utc)

            # Get current time in UTC
            now = datetime.now(timezone.utc)

            length_days = (expiration_date - now).days
            return length_days if length_days >= 0 else 0
        except Exception as e:
            logging.info(
                f"[WHOIS ERROR] domain_registration_length({self.domain}): {e}"
            )
            return -1

    #######################################################################################
    #                          3.3 Domain recognized by WHOIS                             #
    #######################################################################################

    @timer
    @deadline(5)
    def whois_registered_domain(self):
        try:
            hostname = self.res.domain_name
            if type(hostname) == list:
                for host in hostname:
                    if re.search(host.lower(), self.domain):
                        return 0
                return 1
            else:
                if re.search(hostname.lower(), self.domain):
                    return 0
                else:
                    return 1
        except:
            return 1

    #######################################################################################
    #                               3.4 Get web traffic                                   #
    #######################################################################################

    @timer
    def web_traffic(self):
        try:
            rank = BeautifulSoup(
                urlopen(
                    "http://data.alexa.com/data?cli=10&dat=s&url=" + self.url
                ).read(),
                "xml",
            ).find("REACH")["RANK"]
        except:
            return 0
        return int(rank)

    #######################################################################################
    #                                   3.5 Domain age                                    #
    #######################################################################################

    @timer
    @deadline(5)
    def domain_age(self):
        try:
            creation_date = self.res.creation_date

            # Handle case where creation_date is a list
            if isinstance(creation_date, list):
                # Normalize all dates to make them timezone-aware with UTC
                normalized_dates = self.normalize_datetime_list(creation_date)
                if not normalized_dates:
                    return -2  # No valid data
                creation_date = min(normalized_dates)
            elif creation_date is None:
                return -2  # No data
            elif creation_date.tzinfo is None:
                # Single naive datetime
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            else:
                # Single aware datetime but potentially in a different timezone
                creation_date = creation_date.astimezone(timezone.utc)

            # Get current time in UTC
            now = datetime.now(timezone.utc)

            age_days = (now - creation_date).days
            return age_days if age_days >= 0 else -2
        except Exception as e:
            logging.info(f"[WHOIS ERROR] domain_age({self.domain}): {e}")
            return -1

    #######################################################################################
    #                                  3.6 Global rank                                    #
    #######################################################################################

    @timer
    @deadline(3)
    def global_rank(self):
        if self.domain in URL_EXTRACTOR.global_rank_cache:
            return URL_EXTRACTOR.global_rank_cache[self.domain]
        rank_checker_response = requests.post(
            "https://www.checkpagerank.net/index.php", {"name": self.domain}
        )
        try:
            rank = int(
                re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0]
            )
        except:
            rank = -1
        URL_EXTRACTOR.global_rank_cache[self.domain] = rank
        return rank

    #######################################################################################
    #                                 3.7 Google index                                    #
    #######################################################################################

    @timer
    @deadline(5)
    def google_index(self):
        param = {"q": "site:" + self.url}
        google_query = "https://www.google.com/search?" + urlencode(param)
        data = requests.get(google_query, headers=self.headers)
        soup = BeautifulSoup(data.text, "html.parser")
        try:
            if (
                "Our systems have detected unusual traffic from your computer network."
                in soup.text
            ):
                return -1
            # Look for search result links
            rso = soup.find(id="rso")
            if rso:
                links = rso.find_all("a", href=True)
                if links:
                    return 0  # Indexed
            return 1  # Not indexed
        except Exception:
            return 1  # Not indexed or error

    #######################################################################################
    #                        3.8 DNS record expiration length                             #
    #######################################################################################

    @timer
    @deadline(3)
    def dns_record(self):
        if self.domain in URL_EXTRACTOR.dns_record_cache:
            return URL_EXTRACTOR.dns_record_cache[self.domain]
        try:
            nameservers = dns.resolver.query(self.domain, "NS")
            if len(nameservers) > 0:
                result = 0
            else:
                result = 1
        except:
            result = 1
        URL_EXTRACTOR.dns_record_cache[self.domain] = result
        return result

    #######################################################################################
    #                           3.10 Page Rank from OPR                                   #
    #######################################################################################

    @timer
    @deadline(3)
    def page_rank(self):
        if self.domain in URL_EXTRACTOR.page_rank_cache:
            return URL_EXTRACTOR.page_rank_cache[self.domain]
        rank_checker_response = requests.post(
            "https://www.checkpagerank.net/index.php", {"name": self.domain}
        )
        try:
            rank = int(
                re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0]
            )
        except:
            rank = -1
        URL_EXTRACTOR.page_rank_cache[self.domain] = rank
        return rank

    #######################################################################################
    #  _____     __   ____ ___  __  __ ____ ___ _   _ _____   URL's Features: 59          #
    # |_ _\ \   / /  / ___/ _ \|  \/  | __ )_ _| \ | | ____|  Content's Features: 27      #
    #  | | \ \ / /  | |  | | | | |\/| |  _ \| ||  \| |  _|    External Features: 6        #
    #  | |  \ V /   | |__| |_| | |  | | |_) | || |\  | |___   Total Features: 91          #
    # |___|_ \_(_)   \____\___/|_| _|_|____/___|_|_\_|_____|   (label included)           #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    @timer
    def extract_to_dataset(self):
        data = {}
        # List of (key, function) pairs for all features
        feature_funcs = [
            ("url", lambda: self.url),
            ("url_len", self.url_len),
            ("hostname_len", self.hostname_len),
            ("entropy", self.entropy),
            ("nb_fragments", self.count_fragments),
            ("nb_dots", self.count_dots),
            ("nb_hyphens", self.count_hyphens),
            ("nb_at", self.count_at),
            ("nb_exclamation", self.count_exclamation),
            ("nb_and", self.count_and),
            ("nb_or", self.count_or),
            ("nb_equal", self.count_equal),
            ("nb_underscore", self.count_underscore),
            ("nb_tilde", self.count_tilde),
            ("nb_percentage", self.count_percentage),
            ("nb_slash", self.count_slash),
            ("nb_dslash", self.count_double_slash),
            ("nb_star", self.count_star),
            ("nb_colon", self.count_colon),
            ("nb_comma", self.count_comma),
            ("nb_semicolumn", self.count_semicolumn),
            ("nb_dollar", self.count_dollar),
            ("nb_space", self.count_space),
            ("nb_http_token", self.count_http_token),
            ("nb_subdomain", self.count_subdomain),
            ("nb_www", self.count_www),
            ("nb_com", self.count_com),
            ("nb_redirection", self.count_redirection),
            ("nb_e_redirection", self.count_external_redirection),
            ("nb_phish_hints", self.count_phish_hints),
            ("has_ip", self.having_ip_address),
            ("has_https", self.has_https),
            ("has_punnycode", self.has_punycode),
            ("has_port", self.has_port),
            ("has_tld_in_path", self.has_tld_in_path),
            ("has_tld_in_subdomain", self.has_tld_in_subdomain),
            ("has_abnormal_subdomain", self.has_abnormal_subdomain),
            ("has_prefix_suffix", self.has_prefix_suffix),
            ("has_short_svc", self.has_shortening_service),
            ("has_path_txt_extension", self.has_path_txt_extension),
            ("has_path_exe_extension", self.has_path_exe_extension),
            ("has_domain_in_brand", self.has_domain_in_brand),
            ("has_brand_in_path", self.has_brand_in_path),
            ("has_sus_tld", self.has_suspecious_tld),
            ("has_statistical_report", self.has_statistical_report),
            ("word_raw_len", self.length_word_raw),
            ("char_repeat", self.char_repeat),
            ("shortest_word_raw_len", self.shortest_word_raw_length),
            ("shortest_word_raw_host_len", self.shortest_word_raw_host_length),
            ("shortest_word_raw_path_len", self.shortest_word_raw_path_length),
            ("longest_word_raw_len", self.longest_word_raw_length),
            ("longest_word_raw_host_len", self.longest_word_raw_host_length),
            ("longest_word_raw_path_len", self.longest_word_raw_path_length),
            ("avg_word_raw_len", self.average_word_raw_length),
            ("avg_word_raw_host_len", self.average_word_raw_host_length),
            ("avg_word_raw_path_len", self.average_word_raw_path_length),
            ("ratio_digits_url", self.ratio_digits_url),
            ("ratio_digits_host", self.ratio_digits_hostname),
            # Content's Features
            ("is_alive", lambda: self.state),
            ("body_len", self.body_length),
            ("script_len", self.script_length),
            ("empty_title", self.empty_title),
            ("nb_hyperlinks", self.count_hyperlinks),
            ("nb_ex_css", self.count_external_css),
            ("nb_titles", self.nb_titles),
            ("nb_imgs", self.nb_images),
            ("nb_special_char", self.count_special_characters),
            ("has_login_form", self.has_login_form),
            ("has_ex_favicon", self.has_external_favicon),
            ("has_submit_email", self.has_submitting_to_email),
            ("has_iframe", self.has_iframe),
            ("has_onmouse", self.has_on_mouse_action),
            ("has_popup", self.has_popup_window),
            ("has_right_click", self.has_right_click),
            ("has_copyright_domain", self.has_domain_with_copyright),
            ("ratio_in_hyperlinks", self.ratio_internal_hyperlinks),
            ("ratio_ex_hyperlinks", self.ratio_external_hyperlinks),
            ("ratio_script_special_chars", self.ratio_script_to_special_chars),
            ("ratio_script_body", self.ratio_script_to_body),
            ("ratio_body_special_chars", self.ratio_body_to_special_char),
            ("percent_in_media", self.percentile_internal_media),
            ("percent_ex_media", self.percentile_external_media),
            ("percent_safe_anchor", self.percentile_safe_anchor),
            ("percent_in_links", self.percentile_internal_links),
            # External Features
            ("whois_reg_domain", self.whois_registered_domain),
            ("domain_reg_len", self.domain_registration_length),
            ("domain_age", self.domain_age),
            ("dns_record", self.dns_record),
            ("google_index", self.google_index),
            ("page_rank", self.page_rank),
            # Label
            ("label", lambda: self.label),
        ]

        for key, func in tqdm(
            feature_funcs, desc="  Extracting features", unit="feature"
        ):
            data[key] = func()

        return data

    @timer
    def extract_to_predict(self):
        data = {}
        # List of (key, function) pairs for all features
        feature_funcs = [
            ("url", lambda: self.url),
            ("url_len", self.url_len),
            ("hostname_len", self.hostname_len),
            ("entropy", self.entropy),
            ("nb_dots", self.count_dots),
            ("nb_hyphens", self.count_hyphens),
            ("nb_exclamation", self.count_exclamation),
            ("nb_and", self.count_and),
            ("nb_equal", self.count_equal),
            ("nb_underscore", self.count_underscore),
            ("nb_tilde", self.count_tilde),
            ("nb_percentage", self.count_percentage),
            ("nb_slash", self.count_slash),
            ("nb_colon", self.count_colon),
            ("nb_semicolumn", self.count_semicolumn),
            ("nb_space", self.count_space),
            ("nb_http_token", self.count_http_token),
            ("nb_subdomain", self.count_subdomain),
            ("nb_www", self.count_www),
            ("nb_com", self.count_com),
            ("nb_phish_hints", self.count_phish_hints),
            ("has_ip", self.having_ip_address),
            ("has_https", self.has_https),
            ("has_port", self.has_port),
            ("has_tld_in_path", self.has_tld_in_path),
            ("has_tld_in_subdomain", self.has_tld_in_subdomain),
            ("has_abnormal_subdomain", self.has_abnormal_subdomain),
            ("has_prefix_suffix", self.has_prefix_suffix),
            ("has_short_svc", self.has_shortening_service),
            ("has_path_exe_extension", self.has_path_exe_extension),
            ("has_domain_in_brand", self.has_domain_in_brand),
            ("has_sus_tld", self.has_suspecious_tld),
            ("has_statistical_report", self.has_statistical_report),
            ("word_raw_len", self.length_word_raw),
            ("char_repeat", self.char_repeat),
            ("shortest_word_raw_len", self.shortest_word_raw_length),
            ("shortest_word_raw_host_len", self.shortest_word_raw_host_length),
            ("shortest_word_raw_path_len", self.shortest_word_raw_path_length),
            ("longest_word_raw_len", self.longest_word_raw_length),
            ("longest_word_raw_host_len", self.longest_word_raw_host_length),
            ("longest_word_raw_path_len", self.longest_word_raw_path_length),
            ("avg_word_raw_len", self.average_word_raw_length),
            ("avg_word_raw_host_len", self.average_word_raw_host_length),
            ("avg_word_raw_path_len", self.average_word_raw_path_length),
            ("ratio_digits_url", self.ratio_digits_url),
            ("ratio_digits_host", self.ratio_digits_hostname),
            # Content's Features
            ("is_alive", lambda: self.state),
            ("body_len", self.body_length),
            ("script_len", self.script_length),
            ("empty_title", self.empty_title),
            ("nb_hyperlinks", self.count_hyperlinks),
            ("nb_ex_css", self.count_external_css),
            ("nb_titles", self.nb_titles),
            ("nb_imgs", self.nb_images),
            ("nb_special_char", self.count_special_characters),
            ("has_login_form", self.has_login_form),
            ("has_ex_favicon", self.has_external_favicon),
            ("has_copyright_domain", self.has_domain_with_copyright),
            ("ratio_in_hyperlinks", self.ratio_internal_hyperlinks),
            ("ratio_script_special_chars", self.ratio_script_to_special_chars),
            ("ratio_script_body", self.ratio_script_to_body),
            ("ratio_body_special_chars", self.ratio_body_to_special_char),
            ("percent_in_media", self.percentile_internal_media),
            ("percent_ex_media", self.percentile_external_media),
            ("percent_safe_anchor", self.percentile_safe_anchor),
            ("percent_in_links", self.percentile_internal_links),
            # External Features
            ("whois_reg_domain", self.whois_registered_domain),
            ("domain_reg_len", self.domain_registration_length),
            ("domain_age", self.domain_age),
            ("dns_record", self.dns_record),
            ("google_index", self.google_index),
            ("page_rank", self.page_rank),
            # Label
            ("label", lambda: self.label),
        ]

        for key, func in tqdm(
            feature_funcs, desc="  Extracting features", unit="feature"
        ):
            data[key] = func()

        return data