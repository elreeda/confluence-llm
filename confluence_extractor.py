#!/usr/bin/env python3
"""
Confluence Data Extractor for LLM Usage

This script extracts data from Confluence using the API, structures it,
and preprocesses it for use with Large Language Models.
"""

import os
import json
import re
import requests
from bs4 import BeautifulSoup
import html2text
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ConfluencePage:
    """Data class to represent a Confluence page."""
    id: str
    title: str
    space_key: str
    content: str
    url: str
    parent_id: Optional[str] = None
    created_date: Optional[str] = None
    last_updated: Optional[str] = None
    version: Optional[int] = None
    ancestors: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the page to a dictionary."""
        return asdict(self)


class ConfluenceExtractor:
    """Class to extract data from Confluence."""
    
    def __init__(
        self, 
        base_url: str = None, 
        username: str = None, 
        api_token: str = None,
        output_dir: str = "confluence_data"
    ):
        """Initialize the Confluence extractor.
        
        Args:
            base_url: The base URL of the Confluence instance
            username: The username for authentication
            api_token: The API token for authentication
            output_dir: Directory to save extracted data
        """
        self.base_url = base_url or os.getenv("CONFLUENCE_BASE_URL")
        self.username = username or os.getenv("CONFLUENCE_USERNAME")
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN")
        self.output_dir = output_dir
        
        if not all([self.base_url, self.username, self.api_token]):
            raise ValueError(
                "Missing required credentials. Please provide base_url, username, and api_token "
                "either as parameters or as environment variables."
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        
    def get_all_spaces(self) -> List[Dict[str, Any]]:
        """Get all spaces from Confluence.
        
        Returns:
            List of space dictionaries
        """
        logger.info("Fetching all spaces from Confluence")
        spaces = []
        start = 0
        limit = 100
        
        while True:
            url = f"{self.base_url}/rest/api/space?start={start}&limit={limit}"
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch spaces: {response.status_code} - {response.text}")
                raise Exception(f"Failed to fetch spaces: {response.status_code}")
            
            data = response.json()
            spaces.extend(data["results"])
            
            if data["_links"].get("next"):
                start += limit
            else:
                break
                
        logger.info(f"Found {len(spaces)} spaces")
        return spaces
    
    def get_pages_in_space(self, space_key: str) -> List[Dict[str, Any]]:
        """Get all pages in a specific space.
        
        Args:
            space_key: The key of the space
            
        Returns:
            List of page dictionaries
        """
        logger.info(f"Fetching pages for space: {space_key}")
        pages = []
        start = 0
        limit = 100
        
        while True:
            url = f"{self.base_url}/rest/api/content?spaceKey={space_key}&start={start}&limit={limit}&expand=ancestors,version"
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch pages: {response.status_code} - {response.text}")
                raise Exception(f"Failed to fetch pages: {response.status_code}")
            
            data = response.json()
            pages.extend(data["results"])
            
            if data["_links"].get("next"):
                start += limit
            else:
                break
                
        logger.info(f"Found {len(pages)} pages in space {space_key}")
        return pages
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """Get the content of a specific page.
        
        Args:
            page_id: The ID of the page
            
        Returns:
            Page content dictionary
        """
        url = f"{self.base_url}/rest/api/content/{page_id}?expand=body.storage,ancestors,version"
        response = requests.get(
            url,
            auth=(self.username, self.api_token),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch page content: {response.status_code} - {response.text}")
            raise Exception(f"Failed to fetch page content: {response.status_code}")
        
        return response.json()
    
    def preprocess_content(self, html_content: str) -> str:
        """Preprocess HTML content to make it suitable for LLMs.
        
        Args:
            html_content: The HTML content to preprocess
            
        Returns:
            Preprocessed text content
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove common boilerplate elements
        for element in soup.select('.confluence-information-macro, .aui-message'):
            element.decompose()
        
        # Convert to markdown
        markdown = self.html_converter.handle(str(soup))
        
        # Clean up the markdown
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove any remaining HTML tags
        markdown = re.sub(r'<[^>]+>', '', markdown)
        
        return markdown.strip()
    
    def extract_page(self, page_id: str, space_key: str) -> ConfluencePage:
        """Extract a single page and its content.
        
        Args:
            page_id: The ID of the page
            space_key: The key of the space
            
        Returns:
            ConfluencePage object
        """
        page_data = self.get_page_content(page_id)
        
        html_content = page_data["body"]["storage"]["value"]
        processed_content = self.preprocess_content(html_content)
        
        page_url = f"{self.base_url}/display/{space_key}/{page_data['id']}"
        
        ancestors = page_data.get("ancestors", [])
        parent_id = ancestors[-1]["id"] if ancestors else None
        
        return ConfluencePage(
            id=page_data["id"],
            title=page_data["title"],
            space_key=space_key,
            content=processed_content,
            url=page_url,
            parent_id=parent_id,
            created_date=page_data.get("history", {}).get("createdDate"),
            last_updated=page_data.get("history", {}).get("lastUpdated"),
            version=page_data.get("version", {}).get("number"),
            ancestors=ancestors
        )
    
    def extract_all_pages(self, space_keys: Optional[List[str]] = None) -> List[ConfluencePage]:
        """Extract all pages from specified spaces or all spaces.
        
        Args:
            space_keys: List of space keys to extract from. If None, extract from all spaces.
            
        Returns:
            List of ConfluencePage objects
        """
        all_pages = []
        
        # Get all spaces if not specified
        if not space_keys:
            spaces = self.get_all_spaces()
            space_keys = [space["key"] for space in spaces]
        
        for space_key in space_keys:
            logger.info(f"Processing space: {space_key}")
            try:
                pages = self.get_pages_in_space(space_key)
                
                for page in pages:
                    logger.info(f"Processing page: {page['title']} (ID: {page['id']})")
                    try:
                        page_obj = self.extract_page(page["id"], space_key)
                        all_pages.append(page_obj)
                    except Exception as e:
                        logger.error(f"Error processing page {page['id']}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing space {space_key}: {str(e)}")
        
        logger.info(f"Extracted {len(all_pages)} pages in total")
        return all_pages
    
    def save_pages_to_json(self, pages: List[ConfluencePage], filename: str = "confluence_pages.json"):
        """Save extracted pages to a JSON file.
        
        Args:
            pages: List of ConfluencePage objects
            filename: Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([page.to_dict() for page in pages], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(pages)} pages to {output_path}")
    
    def save_pages_to_text_files(self, pages: List[ConfluencePage], directory: Optional[str] = None):
        """Save each page as a separate text file.
        
        Args:
            pages: List of ConfluencePage objects
            directory: Directory to save the files in (within output_dir)
        """
        if directory:
            output_path = os.path.join(self.output_dir, directory)
        else:
            output_path = os.path.join(self.output_dir, "pages")
        
        os.makedirs(output_path, exist_ok=True)
        
        for page in pages:
            # Create a safe filename
            safe_title = re.sub(r'[^\w\s-]', '', page.title).strip().lower()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            filename = f"{safe_title}_{page.id}.md"
            file_path = os.path.join(output_path, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write metadata as YAML front matter
                f.write("---\n")
                f.write(f"title: {page.title}\n")
                f.write(f"id: {page.id}\n")
                f.write(f"space_key: {page.space_key}\n")
                f.write(f"url: {page.url}\n")
                if page.parent_id:
                    f.write(f"parent_id: {page.parent_id}\n")
                if page.last_updated:
                    f.write(f"last_updated: {page.last_updated}\n")
                f.write("---\n\n")
                
                # Write content
                f.write(page.content)
        
        logger.info(f"Saved {len(pages)} pages as text files in {output_path}")


def main():
    """Main function to run the extractor."""
    # Load credentials from environment variables
    load_dotenv()
    
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Extract data from Confluence for LLM usage')
    parser.add_argument('--base-url', help='Confluence base URL')
    parser.add_argument('--username', help='Confluence username')
    parser.add_argument('--api-token', help='Confluence API token')
    parser.add_argument('--output-dir', default='confluence_data', help='Output directory')
    parser.add_argument('--spaces', nargs='+', help='Space keys to extract (if not specified, extract all)')
    parser.add_argument('--format', choices=['json', 'text', 'both'], default='both', 
                        help='Output format (json, text, or both)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the extractor
        extractor = ConfluenceExtractor(
            base_url=args.base_url,
            username=args.username,
            api_token=args.api_token,
            output_dir=args.output_dir
        )
        
        # Extract pages
        pages = extractor.extract_all_pages(args.spaces)
        
        # Save pages in the requested format
        if args.format in ['json', 'both']:
            extractor.save_pages_to_json(pages)
        
        if args.format in ['text', 'both']:
            extractor.save_pages_to_text_files(pages)
            
        logger.info("Extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 