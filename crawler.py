from requests_html import AsyncHTMLSession
import json
import itertools


def url_of_page(pg: int) -> str:
    esrb_url = "https://www.esrb.org/search/?"
    search_filter = [
        "searchKeyword=",
        "platform=All%20Platforms",
        "rating=E%2CE10%2B%2CT%2CM%2CAO",
        "descriptor=All%20Content"
    ]
    search_filter.append(f"pg={pg}")
    search_filter.extend([
        "searchType=LatestRatings",
        "ielement[]=all",
        "timeFrame=All"
    ])

    filter_url = "&".join(search_filter)
    ret = esrb_url + filter_url
    return ret


async def fetch_page(pg: int, asession: AsyncHTMLSession):
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"}
    page = await asession.get(url_of_page(pg), headers=header)
    await page.html.arender(wait=0.2 * (pg % 20), sleep=2.0, timeout=50.0)
    return page


def fetch_pages_up_to(pg_start: int, pg_cnt: int):
    asession = AsyncHTMLSession()
    return asession.run(*[lambda p=p: fetch_page(p, asession) for p in range(pg_start, pg_start + pg_cnt)])


def parse_game(game) -> dict[str, str]:
    title = game.find("div.heading > h2 > a")[0].text
    table = game.find("div.content > table > tbody > tr")[1]
    grade = table.find("td > img")[0].attrs["alt"]
    summary = table.find("td")[3].find("div.synopsis")[0].text
    return {"title": title, "grade": grade, "summary": summary}


def parse_page(page) -> list[dict[str, str]]:
    games = page.html.find("div#results > div.game")
    return list(map(parse_game, games))


def parse_pages(pages) -> list[dict[str, str]]:
    return list(itertools.chain(*map(parse_page, pages)))


if __name__ == "__main__":
    for i in range(1, 4):
        offset = i * 20
        start_pg = offset + 1
        end_pg = offset + 20
        fetched_pages = fetch_pages_up_to(start_pg, end_pg)
        print(f"fetched page {start_pg} - {end_pg}")
        parsed_pages = parse_pages(fetched_pages)
        with open(f"dump{i}.json", "w", encoding="utf-8") as jf:
            json.dump(parsed_pages, jf, ensure_ascii=False)
        print(f"wrote page {start_pg} - {end_pg}")
