import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from util.scraper import Scraper

class CBSACodesScraper(Scraper):

    def scrape(self):
        url = 'https://www.uspto.gov/web/offices/ac/ido/oeip/taf/cls_cbsa/cbsa_countyassoc.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        tables = soup.find_all('table', attrs={'class', 'Table'})
        entries = {'category': [], 'code': [], 'title': [], 'regional_components': []}
        for table in tables:
            for row in table.find_all('tr')[1:]:
                entry = [item.get_text() for item in row.find_all('td', attrs={'class', 'Data'})]
                entries['category'].append(entry[0])
                entries['code'].append(entry[1])
                entries['title'].append(entry[2])
                entries['regional_components'].append(entry[3])

        df = pd.DataFrame(entries)
        df.to_csv(os.path.join(self._path, self._filename), index=False)


def main():
    scraper = CBSACodesScraper(path='../../data/geography/', filename='cbsa_codes.csv')
    scraper.scrape()


if __name__ == '__main__':
    main()