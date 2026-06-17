import time
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

BASE_URL = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "FinSight360 ajoshi8879@sdsu.edu"
REQUEST_DELAY = 0.15


class EdgarClient:
    def __init__(self, user_agent: str = USER_AGENT, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._ticker_map: Optional[dict] = None

    def _get(self, url: str) -> dict:
        time.sleep(self.delay)
        log.debug("edgar_request", url=url)
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    def resolve_cik(self, ticker: str) -> str:
        """Return zero-padded-10-char CIK for the given ticker symbol."""
        if self._ticker_map is None:
            data = self._get(TICKERS_URL)
            self._ticker_map = {
                v["ticker"].upper(): str(v["cik_str"])
                for v in data.values()
            }
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found in SEC tickers list")
        return cik

    def get_submissions(self, cik: str) -> dict:
        """Fetch the submissions JSON for a CIK."""
        padded = cik.zfill(10)
        return self._get(f"{BASE_URL}/submissions/CIK{padded}.json")

    def get_company_facts(self, cik: str) -> dict:
        """Fetch the XBRL company facts JSON for a CIK."""
        padded = cik.zfill(10)
        return self._get(f"{BASE_URL}/api/xbrl/companyfacts/CIK{padded}.json")

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
