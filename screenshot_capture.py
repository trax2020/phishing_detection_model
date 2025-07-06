import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime

class ScreenshotCapture:
    def __init__(self, output_dir="screenshots"):
        """
        Initialize the screenshot capture module.
        
        Args:
            output_dir (str): Directory to save screenshots
        """
        self.output_dir = output_dir
        self._setup_directories()
        self.driver = self._setup_driver()
        
    def _setup_directories(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _setup_driver(self):
        """Setup and return a headless Chrome driver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
        
    def capture_screenshot(self, url):
        """
        Capture a screenshot of the given URL.
        
        Args:
            url (str): URL of the webpage to capture
            
        Returns:
            str: Path to the saved screenshot
        """
        try:
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to load (you might want to add explicit waits here)
            self.driver.implicitly_wait(10)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Take screenshot
            self.driver.save_screenshot(filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Error capturing screenshot: {str(e)}")
            return None
            
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()

# Example usage
if __name__ == "__main__":
    # Example usage
    screenshotter = ScreenshotCapture()
    url = "https://pornhub.com"
    screenshot_path = screenshotter.capture_screenshot(url)
    if screenshot_path:
        print(f"Screenshot saved to: {screenshot_path}")
    screenshotter.close() 