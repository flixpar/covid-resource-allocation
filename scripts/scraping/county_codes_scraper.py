import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from util.scraper import Scraper

class CountyCodesScraper(Scraper):

    def scrape(self):
        url = 'https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table', attrs={'class', 'data'})
        entries = {'code': [], 'name': [], 'state': []}
        for t in table:
            for row in t.find_all('tr')[1:]:
                entry = [item.get_text() for item in row.find_all('td')]
                entries['code'].append(entry[0])
                entries['name'].append(entry[1])
                entries['state'].append(entry[2])

        df = pd.DataFrame(entries)
        print(df.head())
        df.to_csv(os.path.join(self._path, self._filename), index=False)


def main():
    scraper = CountyCodesScraper(path='../../data/geography/', filename='county_codes.csv')
    scraper.scrape()


if __name__ == '__main__':
    main()