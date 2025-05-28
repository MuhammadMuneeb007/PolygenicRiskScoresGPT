import os
import time
import re
import json
import logging
import mimetypes
import hashlib
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeleniumWebsiteDownloader:
    """Class to download complete websites with Selenium for JavaScript-rendered content."""
    
    def __init__(self, max_pages=100, delay=1.0, same_domain_only=True, handle_shtml=True, headless=True):
        """
        Initialize the downloader.
        
        Args:
            max_pages (int): Maximum number of pages to download
            delay (float): Delay between page navigations in seconds
            same_domain_only (bool): Whether to only follow links on the same domain
            handle_shtml (bool): Whether to handle .shtml files specially
            headless (bool): Whether to run Chrome in headless mode
        """
        self.max_pages = max_pages
        self.delay = delay
        self.same_domain_only = same_domain_only
        self.handle_shtml = handle_shtml
        self.headless = headless
        
        self.downloaded_urls = set()
        self.urls_to_visit = deque()
        self.url_to_filename_map = {}
        self.domain = None
        self.driver = None
        self.resource_files = set()
        self.fragment_pages = {}
        self.current_save_dir = None
        
    def _setup_driver(self):
        """Set up the Chrome WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set download preferences
        prefs = {
            "download.default_directory": self.current_save_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Install the latest ChromeDriver and create the driver
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def _get_domain(self, url):
        """Extract the domain from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def _get_absolute_url(self, base_url, href):
        """Convert a relative URL to an absolute URL."""
        return urljoin(base_url, href)
    
    def _is_valid_url(self, url):
        """Check if a URL is valid to be crawled."""
        # Skip URLs that are not http(s)
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Extract URL without fragments for checking visited status
        url_without_fragment, fragment = urldefrag(url)
        
        # Keep track of fragments for each base URL
        if fragment:
            base_url = url_without_fragment
            if base_url not in self.fragment_pages:
                self.fragment_pages[base_url] = set()
            self.fragment_pages[base_url].add(fragment)
            
        # Skip already downloaded URLs
        if url in self.downloaded_urls:
            return False
        
        # Skip URLs that are not on the same domain if same_domain_only is True
        if self.same_domain_only and self._get_domain(url) != self.domain:
            return False
        
        # Skip URLs that are common to avoid
        skip_extensions = ('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
                          '.zip', '.tar', '.gz')
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            self.resource_files.add(url)  # Still track these as resources
            return False
            
        # Handle resource files separately
        resource_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js')
        if any(url.lower().endswith(ext) for ext in resource_extensions):
            self.resource_files.add(url)
            return False
            
        return True
    
    def _create_filename(self, url, save_dir):
        """Create a valid filename from a URL."""
        parsed_url = urlparse(url)
        path = parsed_url.path
        fragment = parsed_url.fragment
        
        # Handle the root URL
        if path == "" or path == "/":
            base_name = "index"
        else:
            # Clean the path for valid filename
            path = re.sub(r'[\\/*?:"<>|]', '_', path)
            path = path.strip('/')
            base_name = path
        
        # Add the fragment identifier to the filename if it exists
        if fragment:
            # Clean the fragment for valid filename
            fragment = re.sub(r'[\\/*?:"<>|=&]', '_', fragment)
            # Add fragment as a suffix to the base filename
            if '.' in base_name.split('/')[-1]:
                # If the filename has an extension, insert the fragment before it
                name_parts = base_name.split('.')
                name_parts[-2] += f"_{fragment}"
                base_name = '.'.join(name_parts)
            else:
                # If no extension, just append the fragment
                base_name += f"_{fragment}"
        
        # Handle query parameters
        if parsed_url.query:
            query_hash = hashlib.md5(parsed_url.query.encode()).hexdigest()[:8]
            if '.' in base_name.split('/')[-1]:
                # If the filename has an extension, insert the hash before it
                name_parts = base_name.split('.')
                name_parts[-2] += f"_q{query_hash}"
                base_name = '.'.join(name_parts)
            else:
                # If no extension, just append the hash
                base_name += f"_q{query_hash}"
        
        # Handle shtml files appropriately - only convert to html if handle_shtml is True
        if self.handle_shtml and base_name.endswith('.shtml'):
            base_name = base_name.replace('.shtml', '.html')
        
        # Ensure the file has an html extension if it doesn't have an extension
        if not base_name.endswith(('.html', '.htm', '.asp', '.php', '.jsp', '.shtml')):
            if '.' not in base_name.split('/')[-1]:
                base_name = base_name + '.html'
        
        # Make sure path parts are valid and not too long
        path_parts = base_name.split('/')
        path_parts = [part[:100] if len(part) > 100 else part for part in path_parts]
        base_name = '/'.join(path_parts)
        
        # Create full path
        full_path = os.path.join(save_dir, base_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        return full_path
    
    def _extract_links_from_page(self, page_url):
        """Extract all links from the current page loaded in Selenium."""
        links = []
        
        try:
            # Wait for page to be interactive
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Let JavaScript execute and render content
            time.sleep(2)
            
            # Get all anchor elements
            a_elements = self.driver.find_elements(By.TAG_NAME, "a")
            
            for a in a_elements:
                try:
                    href = a.get_attribute("href")
                    if href:
                        # Skip JavaScript, mailto, tel links
                        if href.startswith(('javascript:', 'mailto:', 'tel:')):
                            continue
                            
                        # Convert to absolute URL
                        full_url = href  # Already absolute thanks to Selenium
                        
                        # Check if the URL is valid
                        if self._is_valid_url(full_url):
                            links.append(full_url)
                except StaleElementReferenceException:
                    # Element might have become stale if the page updated
                    continue
                except Exception as e:
                    logger.warning(f"Error extracting href from anchor: {e}")
            
            # Also collect resources (images, CSS, JS)
            for tag_name, attr in [("img", "src"), ("link", "href"), ("script", "src")]:
                elements = self.driver.find_elements(By.TAG_NAME, tag_name)
                for elem in elements:
                    try:
                        attr_value = elem.get_attribute(attr)
                        if attr_value and attr_value.startswith(('http://', 'https://')):
                            self.resource_files.add(attr_value)
                    except:
                        continue
                        
            logger.info(f"Extracted {len(links)} links from {page_url}")
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links from {page_url}: {e}")
            return []
    
    def _save_link_file(self, url, html_filename):
        """Save the original URL to a txt file with the same base name as the HTML file."""
        # Get the base path without extension
        base_path = os.path.splitext(html_filename)[0]
        link_filename = f"{base_path}_page_link.txt"
        
        # Save the URL to the text file
        try:
            with open(link_filename, 'w', encoding='utf-8') as f:
                f.write(url)
            logger.info(f"Saved link file: {link_filename}")
            return link_filename
        except Exception as e:
            logger.error(f"Error saving link file for {url}: {e}")
            return None
    
    def _navigate_to_and_download_page(self, url, save_dir):
        """Navigate to a URL with Selenium and save the page."""
        try:
            logger.info(f"Navigating to: {url}")
            
            # Check if we've already downloaded this page
            if url in self.downloaded_urls:
                return []
                
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to be interactive
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                # Additional wait for JavaScript rendering
                time.sleep(2)
            except TimeoutException:
                logger.warning(f"Timeout waiting for page to load: {url}")
            
            # Create filename
            filename = self._create_filename(url, save_dir)
            self.url_to_filename_map[url] = filename
            
            # Save the current page source
            page_source = self.driver.page_source
            
            # Create BeautifulSoup object to modify URLs
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Fix URLs in the HTML to be absolute
            self._fix_urls_in_html(soup, url)
            
            # Save the HTML file
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            
            # Save the page link to a separate text file
            self._save_link_file(url, filename)
            
            logger.info(f"Downloaded: {url} -> {filename}")
            
            # Mark as downloaded
            self.downloaded_urls.add(url)
            
            # Extract links from the page
            links = self._extract_links_from_page(url)
            
            # Taking screenshots for visualization
            screenshot_dir = os.path.join(save_dir, "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_file = os.path.join(screenshot_dir, os.path.basename(filename).replace('.html', '.png'))
            try:
                #self.driver.save_screenshot(screenshot_file)
                logger.info(f"Saved screenshot: {screenshot_file}")
            except Exception as e:
                logger.warning(f"Failed to save screenshot: {e}")
                
            return links
            
        except WebDriverException as e:
            logger.error(f"WebDriver error navigating to {url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return []
    
    def _fix_urls_in_html(self, soup, base_url):
        """Fix URLs in HTML to be absolute for reliable navigation."""
        # Fix a tags (links)
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip javascript, mailto, tel links
            if href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            full_url = urljoin(base_url, href)
            a_tag['href'] = full_url
            
            # Add data attribute for tracking original URL
            a_tag['data-original-url'] = href
        
        # Fix img tags, link tags, and script tags
        for tag in soup.find_all(['img', 'link', 'script']):
            attr = 'src' if tag.name in ['img', 'script'] else 'href'
            if tag.get(attr):
                src = tag[attr]
                # Make URL absolute
                full_url = urljoin(base_url, src)
                tag[attr] = full_url
                # Add data attribute for tracking original URL
                tag['data-original-url'] = src
    
    def _download_resource(self, url, save_dir):
        return
        """Download a resource file such as image, CSS, or JavaScript."""
        try:
            # Remove URL fragments
            url, _ = urldefrag(url)
            
            # Skip if already downloaded
            if url in self.downloaded_urls:
                return
            
            # Skip if not from the same domain (if that restriction is enabled)
            if self.same_domain_only and self._get_domain(url) != self.domain:
                return
                
            # Use requests for resources to avoid Selenium overhead
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            response = requests.get(url, timeout=15, headers=headers)
            if response.status_code != 200:
                logger.warning(f"Failed to download resource: {url}, status code: {response.status_code}")
                return
                
            # Get the path from the URL
            parsed_url = urlparse(url)
            path = parsed_url.path.strip('/')
            
            # If there's no extension, try to guess it from the content
            if '.' not in path.split('/')[-1]:
                content_type = response.headers.get('Content-Type', '')
                ext = mimetypes.guess_extension(content_type.split(';')[0].strip())
                if ext:
                    path += ext
                    
            # Clean the path
            path = re.sub(r'[\\/*?:"<>|]', '_', path)
            
            # Create the resources directory
            resource_dir = os.path.join(save_dir, 'resources')
            os.makedirs(resource_dir, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(resource_dir, path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Downloaded resource: {url} -> {file_path}")
            self.downloaded_urls.add(url)
            
        except Exception as e:
            logger.error(f"Error downloading resource {url}: {e}")
    
    def download_website(self, website_url, save_dir):
        """Download a complete website starting from a URL using Selenium."""
        # Reset state for a new download
        self.downloaded_urls.clear()
        self.urls_to_visit.clear()
        self.url_to_filename_map.clear()
        self.resource_files.clear()
        self.fragment_pages.clear()
        self.current_save_dir = save_dir
        
        # Set the domain
        self.domain = self._get_domain(website_url)
        if not self.domain:
            logger.error(f"Invalid URL: {website_url}")
            return False
        
        # Create main directory for the website
        website_dir = save_dir
        os.makedirs(website_dir, exist_ok=True)
        
        # Initialize the WebDriver
        try:
            self._setup_driver()
        except Exception as e:
            logger.error(f"Failed to set up WebDriver: {e}")
            return False
        
        try:
            # Add the starting URL
            self.urls_to_visit.append(website_url)
            
            # Keep a file with metadata
            metadata_path = os.path.join(website_dir, "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
                metadata_file.write(f"Website URL: {website_url}\n")
                metadata_file.write(f"Download started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                metadata_file.write(f"Domain: {self.domain}\n")
                metadata_file.write("Downloaded pages:\n")
                
                # Breadth-first crawl of the website
                page_count = 0
                visited_url_patterns = set()  # Track URL patterns to avoid similar pages
                
                while self.urls_to_visit and page_count < self.max_pages:
                    current_url = self.urls_to_visit.popleft()
                    
                    # Skip if already downloaded
                    if current_url in self.downloaded_urls:
                        continue
                    
                    # Skip similar URL patterns (helps avoid pagination traps)
                    url_pattern = self._get_url_pattern(current_url)
                    if len(visited_url_patterns) > 50 and url_pattern in visited_url_patterns:
                        # If we've seen many similar URLs, start skipping
                        if len([p for p in visited_url_patterns if p == url_pattern]) > 5:
                            logger.info(f"Skipping potentially repetitive URL: {current_url}")
                            continue
                    visited_url_patterns.add(url_pattern)
                    
                    # Navigate to and download the page
                    new_links = self._navigate_to_and_download_page(current_url, website_dir)
                    
                    page_count += 1
                    
                    # Log in metadata file
                    metadata_file.write(f"{page_count}. {current_url} -> {self._create_filename(current_url, website_dir)}\n")
                    
                    # Add new links to queue
                    for link in new_links:
                        if link not in self.downloaded_urls and link not in self.urls_to_visit:
                            self.urls_to_visit.append(link)
                    
                    # Respect the delay between requests
                    time.sleep(self.delay)
                    
                    # Periodically save progress
                    if page_count % 10 == 0:
                        logger.info(f"Progress: Downloaded {page_count} pages, {len(self.urls_to_visit)} URLs in queue")
                
                # Save fragment information
                if self.fragment_pages:
                    fragment_info_path = os.path.join(website_dir, "fragment_info.json")
                    with open(fragment_info_path, 'w', encoding='utf-8') as f:
                        fragment_info = {url: list(fragments) for url, fragments in self.fragment_pages.items()}
                        json.dump(fragment_info, f, indent=2)
                    logger.info(f"Saved fragment information to {fragment_info_path}")
                
                # Download all discovered resource files
                metadata_file.write("\nDownloaded resources:\n")
                resource_count = 0
                
                for resource_url in self.resource_files:
                    # Skip if already downloaded
                    if resource_url in self.downloaded_urls:
                        continue
                        
                    self._download_resource(resource_url, website_dir)
                    resource_count += 1
                    metadata_file.write(f"{resource_count}. {resource_url}\n")
                    time.sleep(self.delay * 0.5)  # Use shorter delay for resources
                
                metadata_file.write(f"\nDownload completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                metadata_file.write(f"Total pages downloaded: {page_count}\n")
                metadata_file.write(f"Total resources downloaded: {resource_count}\n")
                
                logger.info(f"Website download complete. Downloaded {page_count} pages and {resource_count} resources to {website_dir}")
                return True
                
        except Exception as e:
            logger.error(f"Error during website download: {e}")
            return False
        finally:
            # Always close the driver when done
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver closed")
    
    def _get_url_pattern(self, url):
        """Generate a pattern from a URL to identify similar URLs (like pagination)."""
        # Remove query parameters and fragments
        parsed = urlparse(url)
        path = parsed.path
        
        # Replace numeric segments with {N} to identify pagination patterns
        pattern = re.sub(r'/\d+/', '/{N}/', path)
        pattern = re.sub(r'/page/\d+', '/page/{N}', pattern)
        pattern = re.sub(r'/p/\d+', '/p/{N}', pattern)
        
        return pattern

def _create_interactive_clickable_map(downloaded_pages, output_dir):
    """Create an interactive HTML map of all downloaded pages with clickable links."""
    map_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Website Map</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .site-map { max-width: 1200px; margin: 0 auto; }
            .page-entry { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            .page-title { font-weight: bold; }
            .screenshot { max-width: 300px; border: 1px solid #ccc; margin-top: 10px; }
            .links-list { margin-top: 10px; }
            .section { margin-bottom: 30px; }
            h2 { border-bottom: 1px solid #eee; padding-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="site-map">
            <h1>Website Crawl Map</h1>
            <div class="section">
                <h2>Pages Downloaded</h2>
    """
    
    # Add each downloaded page
    for url, filename in downloaded_pages.items():
        # Get the link file path
        link_file = f"{os.path.splitext(filename)[0]}_page_link.txt"
        link_file_relative = os.path.relpath(link_file, output_dir) if os.path.exists(link_file) else ""
        
        page_entry = f"""
        <div class="page-entry">
            <div class="page-title">{url}</div>
            <div>Saved as: {os.path.basename(filename)}</div>
            <a href="{os.path.relpath(filename, output_dir)}" target="_blank">View Page</a>
            {f'<a href="{link_file_relative}" target="_blank" style="margin-left:10px;">View Link File</a>' if link_file_relative else ''}
            
            <!-- Screenshot if available -->
            <div>
                <img class="screenshot" src="screenshots/{os.path.basename(filename).replace('.html', '.png')}" 
                     onerror="this.style.display='none'" alt="Page Screenshot">
            </div>
        </div>
        """
        map_html += page_entry
    
    # Close HTML
    map_html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the map
    map_path = os.path.join(output_dir, "website_map.html")
    with open(map_path, 'w', encoding='utf-8') as f:
        f.write(map_html)
    
    logger.info(f"Created interactive website map: {map_path}")
    return map_path

def download_website(website_url, save_dir, max_pages=1000, headless=True):
    """Download a website with all linked pages using Selenium."""
    if not website_url.startswith(("http://", "https://")):
        logger.error(f"Invalid Website URL: {website_url}")
        return False
    
    try:
        # Create the Website subdirectory
        website_dir = os.path.join(save_dir, "Website")
        os.makedirs(website_dir, exist_ok=True)
        
        # Initialize and run the downloader
        downloader = SeleniumWebsiteDownloader(
            max_pages=max_pages, 
            delay=1.0, 
            same_domain_only=True, 
            handle_shtml=True,
            headless=headless
        )
        
        success = downloader.download_website(website_url, website_dir)
        
        # Create an interactive map of downloaded pages
        if success and downloader.url_to_filename_map:
            _create_interactive_clickable_map(downloader.url_to_filename_map, website_dir)
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to download website {website_url}: {e}")
        
        # Create a basic error report
        website_dir = os.path.join(save_dir, "Website")
        os.makedirs(website_dir, exist_ok=True)
        
        with open(os.path.join(website_dir, "error_log.txt"), "w", encoding="utf-8") as error_file:
            error_file.write(f"Failed to download website: {website_url}\n")
            error_file.write(f"Error: {str(e)}\n")
            error_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download a complete website using Selenium.")
    parser.add_argument("url", help="URL of the website to download")
    parser.add_argument("--output", "-o", default="./download", help="Output directory")
    parser.add_argument("--max-pages", "-m", type=int, default=100, help="Maximum number of pages to download")
    parser.add_argument("--visible", "-v", action="store_true", help="Run Chrome in visible mode (not headless)")
    
    args = parser.parse_args()
    
    download_website(args.url, args.output, args.max_pages, not args.visible)