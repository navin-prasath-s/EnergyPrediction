import os
import time
from datetime import datetime, timedelta
import tempfile
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class PJMDataFetcher:
    def __init__(self, download_dir: str = None, headless: bool = True):
        self.download_dir = download_dir or tempfile.mkdtemp()
        self.headless = headless
        self.download_wait = 30
        self.driver = self._setup_driver()

        os.makedirs(self.download_dir, exist_ok=True)

    def _setup_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.headless:
            options.add_argument('--headless=new')
        prefs = {'download.default_directory': self.download_dir}
        options.add_experimental_option('prefs', prefs)
        return webdriver.Chrome(options=options)


    def _wait_for_file_download(self, before: set, timeout: int = 60) -> str:
        start = time.time()
        while True:
            files = set(os.listdir(self.download_dir))
            new_files = files - before
            finished = [f for f in new_files if not f.endswith('.crdownload')]
            if finished:
                return os.path.join(self.download_dir, finished[0])
            if time.time() - start > timeout:
                raise TimeoutError("Download timed out.")
            time.sleep(1)


    def _download_via_xpath(self, xpath: str, url: str) -> pd.DataFrame:
        self.driver.get(url)
        time.sleep(5)  # Wait for page load
        before = set(os.listdir(self.download_dir))
        self.driver.find_element(By.XPATH, xpath).click()
        file_path = self._wait_for_file_download(before, timeout=self.download_wait)
        df = pd.read_csv(file_path)
        os.remove(file_path)
        return df


    def _force_fill_time(self, input_element, value: str):
        """Sets input value via JS and triggers Angular-compatible events."""
        self.driver.execute_script("""
            const el = arguments[0];
            const val = arguments[1];
            el.value = val;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            el.dispatchEvent(new Event('blur', { bubbles: true }));
            el.dispatchEvent(new Event('keyup', { bubbles: true }));
        """, input_element, value)


    def fetch_recent_actual(self) -> pd.DataFrame:
        """Fetches recent actual instantaneous load data."""

        actual = '/html/body/app-root/main/dm-feed-main/div/div/div/section/dm-feed-list/div[2]/main/dm-frequently-accessed/div/div[2]/div[3]/div[2]/ul/li/span[2]/span[3]/dm-feed-download-link/a'
        url = 'https://dataminer2.pjm.com/list'

        return self._download_via_xpath(actual, url)


    def fetch_recent_forecast(self) -> pd.DataFrame:
        """Clicks feed download for latest forecast"""

        forecast = '/html/body/app-root/main/dm-feed-main/div/div/div/section/dm-feed-list/div[2]/main/dm-frequently-accessed/div/div[2]/div[4]/div[2]/ul/li[3]/span[2]/span[3]/dm-feed-download-link/a'
        url = 'https://dataminer2.pjm.com/list'

        return self._download_via_xpath(forecast, url)

    def fetch_historical_actual(self) -> pd.DataFrame:
        """
        Fills 6-hour historical time window, submits, waits for paginator,
        exports CSV, and returns as DataFrame.
        """

        url = 'https://dataminer2.pjm.com/feed/inst_load'
        self.driver.get(url)

        # Wait for inputs to load
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[date-format="M/D/YYYY"]'))
        )

        # Locate input elements
        date_inputs = self.driver.find_elements(By.CSS_SELECTOR, 'input[date-format="M/D/YYYY"]')
        time_inputs = self.driver.find_elements(By.CSS_SELECTOR, 'input[type="text"]')

        start_date_input = date_inputs[0]
        end_date_input = date_inputs[1]
        start_time_input = time_inputs[0]
        end_time_input = time_inputs[1]

        # Read current end datetime from page
        end_date_str = end_date_input.get_attribute("value")  # e.g. '6/10/2025'
        end_time_str = end_time_input.get_attribute("value")  # e.g. '14:55'
        end_dt = datetime.strptime(f"{end_date_str} {end_time_str}", "%m/%d/%Y %H:%M")

        # Compute start datetime (6 hours earlier, rounded to nearest 5 minutes)
        start_dt = end_dt - timedelta(hours=6)
        start_dt = start_dt.replace(minute=(start_dt.minute // 5) * 5, second=0, microsecond=0)

        # Format for input
        start_date_str = f"{start_dt.month}/{start_dt.day}/{start_dt.year}"
        start_time_str = start_dt.strftime("%H:%M")

        # Fill form using JS to trigger Angular model updates
        self._force_fill_time(start_time_input, start_time_str)
        self._force_fill_time(start_date_input, start_date_str)

        # Trigger Angular to acknowledge form changes
        end_date_input.click()  # defocus start_time_input

        print(f"Filled form from {start_date_str} {start_time_str} to {end_date_str} {end_time_str}")

        # Submit
        submit_button = self.driver.find_element(By.XPATH, '//button[contains(text(), "Submit")]')
        submit_button.click()

        time.sleep(2)

        # Now click Export
        export_button = WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.dm-download'))
        )
        before = set(os.listdir(self.download_dir))
        export_button.click()

        file_path = self._wait_for_file_download(before, timeout=self.download_wait)
        df = pd.read_csv(file_path)
        os.remove(file_path)
        return df