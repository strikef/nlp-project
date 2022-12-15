import json
import csv

if __name__ == "__main__":
    review_table = []
    for i in range(1, 4):
        game_reviews = []
        with open(f"dump{i}.json", "r", encoding="utf-8") as jf:
            game_reviews = json.load(jf)
        review_table.extend(list(map(lambda i_rv: [i_rv[0], i_rv[1]["summary"], i_rv[1]["grade"]], enumerate(
            filter(lambda rv: rv["summary"] != "No Rating Summary", game_reviews)))))
    
    with open("reviews.csv", "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(["index", "summary", "grade"])
        cw.writerows(review_table)
