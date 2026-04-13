#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from typing import Dict, Any, List, Optional

import requests
import pandas as pd

DEFAULT_BASE_URL = "https://tripadvisor-scraper-api.omkar.cloud/tripadvisor"


def parse_entity_id_from_link(link: str) -> Optional[int]:
    if not link:
        return None
    m = re.search(r"-d(\d+)-", link)
    return int(m.group(1)) if m else None


class TripAdvisorScraperClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 40, max_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"API-Key": api_key})

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err = None

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                sleep_s = min(2 ** attempt, 12)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 429:
                sleep_s = min(2 ** attempt, 20)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 401:
                raise RuntimeError("401 Unauthorized: Invalid API_KEY or not passed correctly")
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} request failed: {resp.text[:300]}")

            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("API response is not a JSON object")
            return data

        if last_err:
            raise RuntimeError(f"Network request failed (failed after retries): {last_err}")
        raise RuntimeError("Request failed (failed after retries)")

    def restaurants_search(self, query: str, locale: str = "en-US") -> Dict[str, Any]:
        return self._get("/restaurants/search", {"query": query, "locale": locale})

    def restaurants_list(
        self,
        query: str,
        page: int = 1,
        locale: str = "en-US",
        currency: str = "HKD",
        min_rating: Optional[str] = None,
        establishment_types: Optional[str] = "restaurants",
    ) -> Dict[str, Any]:
        params = {"query": query, "page": page, "locale": locale}
        if min_rating:
            params["min_rating"] = min_rating
        if establishment_types:
            params["establishment_types"] = establishment_types
        return self._get("/restaurants/list", params)

    def reviews(
        self,
        query: str,
        page: int = 1,
        locale: str = "en-US",
        lang: Optional[str] = None,
        rating_is: Optional[str] = None,
        since: Optional[str] = None,
        traveler_type: Optional[str] = None,
        keyword: Optional[str] = None,
        sort_by: str = "most_recent",
    ) -> Dict[str, Any]:
        params = {"query": query, "page": page, "locale": locale, "sort_by": sort_by}
        if lang:
            params["lang"] = lang
        if rating_is:
            params["rating_is"] = rating_is
        if since:
            params["since"] = since
        if traveler_type:
            params["traveler_type"] = traveler_type
        if keyword:
            params["keyword"] = keyword
        return self._get("/reviews", params)


def pick_hong_kong_entity_id(search_data: Dict[str, Any]) -> Optional[int]:
    results = search_data.get("results", []) or []
    # Prioritize Hong Kong city
    for r in results:
        name = (r.get("name") or "").lower()
        place_type = (r.get("place_type") or "").upper()
        eid = r.get("tripadvisor_entity_id")
        if "hong kong" in name and place_type in {"CITY", "STATE", "REGION"} and eid:
            return int(eid)
    # Fallback to first result with entity_id
    for r in results:
        eid = r.get("tripadvisor_entity_id")
        if eid:
            return int(eid)
    return None


def normalize_restaurant_row(item: Dict[str, Any]) -> Dict[str, Any]:
    addr = item.get("address") if isinstance(item.get("address"), dict) else {}
    parent = item.get("parent_location") if isinstance(item.get("parent_location"), dict) else {}
    coord = item.get("coordinates") if isinstance(item.get("coordinates"), dict) else {}

    link = item.get("link")
    entity_id = item.get("tripadvisor_entity_id") or parse_entity_id_from_link(link)

    return {
        "tripadvisor_entity_id": entity_id,
        "name": item.get("name"),
        "link": link,
        "rating": item.get("rating"),
        "reviews_count": item.get("reviews"),
        "price_range": item.get("price_range"),
        "phone": item.get("phone"),
        "is_open_now": item.get("is_open_now"),
        "status_text": item.get("status_text"),
        "address": addr.get("address"),
        "city": addr.get("city"),
        "postal_code": addr.get("postal_code"),
        "country": addr.get("country"),
        "country_code": addr.get("country_code"),
        "latitude": coord.get("latitude"),
        "longitude": coord.get("longitude"),
        "parent_location_id": parent.get("tripadvisor_entity_id"),
        "parent_location_name": parent.get("name"),
        "featured_image": item.get("featured_image"),
        "has_reservation": item.get("has_reservation"),
        "has_delivery": item.get("has_delivery"),
        "is_ad": item.get("is_ad"),
    }


def normalize_review_row(review: Dict[str, Any], rest: Dict[str, Any]) -> Dict[str, Any]:
    trip = review.get("trip") if isinstance(review.get("trip"), dict) else {}
    reviewer = review.get("reviewer") if isinstance(review.get("reviewer"), dict) else {}
    hometown = reviewer.get("hometown") if isinstance(reviewer.get("hometown"), dict) else {}

    return {
        "restaurant_entity_id": rest.get("tripadvisor_entity_id"),
        "restaurant_name": rest.get("name"),
        "restaurant_link": rest.get("link"),
        "review_id": review.get("review_id"),
        "review_link": review.get("review_link"),
        "title": review.get("title"),
        "text": review.get("text"),
        "rating": review.get("rating"),
        "language": review.get("language"),
        "original_language": review.get("original_language"),
        "like_count": review.get("like_count"),
        "trip_type": trip.get("trip_type"),
        "stay_date": trip.get("stay_date"),
        "created_at_date": review.get("created_at_date"),
        "published_at_date": review.get("published_at_date"),
        "reviewer_id": reviewer.get("reviewer_id"),
        "reviewer_name": reviewer.get("name"),
        "reviewer_username": reviewer.get("username"),
        "reviewer_profile_link": reviewer.get("profile_link"),
        "reviewer_contribution_count": reviewer.get("contribution_count"),
        "reviewer_hometown": hometown.get("location_name_detailed"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Hong Kong")
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--currency", default="HKD")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--min-rating", default=None)
    parser.add_argument("--max-rest-pages", type=int, default=10)
    parser.add_argument("--max-review-pages", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.6)
    parser.add_argument("--since", default=None)
    parser.add_argument("--rating-is", default=None)
    parser.add_argument("--traveler-type", default=None)
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--sort-by", default="most_recent", choices=["most_recent", "detailed_reviews"])
    parser.add_argument("--restaurants-csv", default="data/raw/hongkong_restaurants.csv")
    parser.add_argument("--reviews-csv", default="data/raw/hongkong_restaurant_reviews.csv")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Please set API_KEY environment variable first")

    client = TripAdvisorScraperClient(api_key=api_key, base_url=os.getenv("BASE_URL", DEFAULT_BASE_URL))

    print("[0/3] Parsing Hong Kong TripAdvisor entity_id...")
    entity_id = None
    try:
        sd = client.restaurants_search(query=args.query, locale=args.locale)
        entity_id = pick_hong_kong_entity_id(sd)
    except Exception as e:
        print(f"  Search failed, attempting fallback: {e}")

    if not entity_id:
        entity_id = 294217  # Hong Kong default ID
        print(f"  Entity ID not found, using fallback: {entity_id}")
    else:
        print(f"  Entity ID found: {entity_id}")

    all_restaurants, all_reviews = [], []
    seen_rest, seen_review = set(), set()

    print(f"[1/3] Fetching restaurant list pages 1...{args.max_rest_pages}")
    for page in range(1, args.max_rest_pages + 1):
        data = client.restaurants_list(
            query=str(entity_id),
            page=page,
            locale=args.locale,
            currency=args.currency,
            min_rating=args.min_rating,
            establishment_types="restaurants",
        )
        results = data.get("results", []) or []
        if not results:
            print(f"  - Page {page} has no results, stopping")
            break

        print(f"  - Page {page}: {len(results)} restaurants")
        for item in results:
            row = normalize_restaurant_row(item)
            k = row.get("tripadvisor_entity_id") or row.get("link")
            if not k or k in seen_rest:
                continue
            seen_rest.add(k)
            all_restaurants.append(row)
        time.sleep(args.sleep)

    if not all_restaurants:
        raise SystemExit("No restaurants found. Check API_KEY quota or reduce --max-rest-pages for testing.")

    print(f"[2/3] Fetching reviews: {len(all_restaurants)} restaurants, max {args.max_review_pages} pages each")
    for i, rest in enumerate(all_restaurants, 1):
        rest_query = rest.get("tripadvisor_entity_id") or rest.get("link") or rest.get("name")
        if not rest_query:
            continue

        print(f"  - ({i}/{len(all_restaurants)}) {rest.get('name')} | query={rest_query}")
        for rp in range(1, args.max_review_pages + 1):
            try:
                rv = client.reviews(
                    query=str(rest_query),
                    page=rp,
                    locale=args.locale,
                    lang=args.lang,
                    rating_is=args.rating_is,
                    since=args.since,
                    traveler_type=args.traveler_type,
                    keyword=args.keyword,
                    sort_by=args.sort_by,
                )
            except Exception as e:
                print(f"    Review page {rp} failed: {e}")
                break

            rlist = rv.get("results", []) or []
            if not rlist:
                break

            print(f"    Review page {rp}: {len(rlist)} reviews")
            for r in rlist:
                rk = r.get("review_id") or r.get("review_link")
                if not rk or rk in seen_review:
                    continue
                seen_review.add(rk)
                all_reviews.append(normalize_review_row(r, rest))
            time.sleep(args.sleep)

    print("[3/3] Writing to CSV")
    pd.DataFrame(all_restaurants).to_csv(args.restaurants_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_reviews).to_csv(args.reviews_csv, index=False, encoding="utf-8-sig")
    print(f"✅ {args.restaurants_csv}: {len(all_restaurants)} rows")
    print(f"✅ {args.reviews_csv}: {len(all_reviews)} rows")


if __name__ == "__main__":
    main()
