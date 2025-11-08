"""
Crawl Service - Web crawling and content extraction
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class CrawlService:
    """Service for crawling web pages"""
    
    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 100,
        respect_robots: bool = True,
        timeout: int = 30
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self.timeout = timeout
        self.user_agent = "RAG-Explorer-Bot/1.0"
    
    def crawl_single_page(self, url: str) -> Dict[str, Any]:
        """Crawl a single page and extract content"""
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            text_content = self._extract_text(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return {
                "url": url,
                "content": text_content,
                "metadata": metadata,
                "status": "success",
                "status_code": response.status_code,
                "crawled_at": datetime.utcnow().isoformat()
            }
            
        except requests.RequestException as e:
            logger.error(f"Error crawling {url}: {e}")
            return {
                "url": url,
                "content": "",
                "metadata": {},
                "status": "error",
                "error": str(e),
                "crawled_at": datetime.utcnow().isoformat()
            }
    
    def crawl_sitemap(self, sitemap_url: str) -> List[Dict[str, Any]]:
        """Crawl all URLs from a sitemap"""
        try:
            response = requests.get(sitemap_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]
            
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
            results = []
            for url in urls[:self.max_pages]:
                result = self.crawl_single_page(url)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error crawling sitemap: {e}")
            return []
    
    def crawl_recursive(
        self,
        start_url: str,
        allowed_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Recursively crawl website starting from URL"""
        visited: Set[str] = set()
        to_visit: List[tuple] = [(start_url, 0)]  # (url, depth)
        results = []
        
        # Determine allowed domains
        if allowed_domains is None:
            parsed = urlparse(start_url)
            allowed_domains = [parsed.netloc]
        
        while to_visit and len(visited) < self.max_pages:
            url, depth = to_visit.pop(0)
            
            # Skip if already visited or max depth exceeded
            if url in visited or depth > self.max_depth:
                continue
            
            # Crawl page
            result = self.crawl_single_page(url)
            visited.add(url)
            results.append(result)
            
            # Extract links if crawl was successful
            if result["status"] == "success" and depth < self.max_depth:
                try:
                    response = requests.get(url, timeout=self.timeout)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = self._extract_links(soup, url, allowed_domains)
                    
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
                            
                except Exception as e:
                    logger.error(f"Error extracting links from {url}: {e}")
        
        logger.info(f"Crawled {len(results)} pages")
        return results
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {"source_url": url}
        
        # Title
        if soup.title:
            metadata["title"] = soup.title.string
        
        # Meta tags
        meta_tags = {
            "description": soup.find("meta", attrs={"name": "description"}),
            "keywords": soup.find("meta", attrs={"name": "keywords"}),
            "author": soup.find("meta", attrs={"name": "author"}),
        }
        
        for key, tag in meta_tags.items():
            if tag and tag.get("content"):
                metadata[key] = tag.get("content")
        
        # Open Graph tags
        og_tags = soup.find_all("meta", attrs={"property": re.compile(r"^og:")})
        for tag in og_tags:
            prop = tag.get("property")
            content = tag.get("content")
            if prop and content:
                metadata[prop] = content
        
        # Headings
        headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2'])]
        if headings:
            metadata["headings"] = headings[:5]  # First 5 headings
        
        return metadata
    
    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
        allowed_domains: List[str]
    ) -> List[str]:
        """Extract valid links from page"""
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Parse URL
            parsed = urlparse(absolute_url)
            
            # Filter criteria
            if (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc in allowed_domains and
                not self._is_excluded_url(absolute_url)
            ):
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def _is_excluded_url(self, url: str) -> bool:
        """Check if URL should be excluded"""
        excluded_extensions = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip',
            '.exe', '.dmg', '.mp4', '.mp3', '.css', '.js'
        ]
        
        url_lower = url.lower()
        return any(url_lower.endswith(ext) for ext in excluded_extensions)
    
    def check_robots_txt(self, base_url: str) -> Dict[str, Any]:
        """Check robots.txt for crawling rules"""
        try:
            parsed = urlparse(base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                return {
                    "exists": True,
                    "content": response.text
                }
            else:
                return {"exists": False}
                
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            return {"exists": False, "error": str(e)}


# Create singleton instance
crawl_service = CrawlService()
