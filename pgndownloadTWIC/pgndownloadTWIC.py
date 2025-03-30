import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO

# URL of the TWIC archive page
twic_url = 'https://theweekinchess.com/twic'

# Directory to save the downloaded and extracted files
output_dir = 'twic_pgn_files'
os.makedirs(output_dir, exist_ok=True)

def get_pgn_zip_links(url):
    """
    Fetches the TWIC archive page and extracts all hyperlinks to PGN zip files.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.zip') and 'twic' in href.lower() and 'g' in href.lower():
            if not href.startswith('http'):
                href = f"https://theweekinchess.com{href}"
            links.append(href)
    return links

def download_and_extract_zip(url, output_folder):
    """
    Downloads a zip file from the given URL and extracts its contents into the specified folder.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(output_folder)

def main():
    # Step 1: Get all PGN zip file links from the TWIC archive page
    pgn_zip_links = get_pgn_zip_links(twic_url)
    print(f'Found {len(pgn_zip_links)} PGN zip files.')

    # Step 2: Download and extract each zip file
    for idx, link in enumerate(pgn_zip_links, start=1):
        print(f'Downloading and extracting file {idx}/{len(pgn_zip_links)}: {link}')
        try:
            download_and_extract_zip(link, output_dir)
            print(f'Successfully extracted: {link}')
        except Exception as e:
            print(f'Failed to process {link}: {e}')
    
    for filename in os.listdir(output_dir):
        if not filename.endswith('.pgn'):
            os.remove(os.path.join(output_dir, filename))
            print(f'Removed non-PGN file: {filename}')

if __name__ == '__main__':
    main()
