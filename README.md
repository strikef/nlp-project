## Content-Based Automated Game Rating Model

### How to run the model
1. Run setup-venv.sh to configure dependencies in a venv
```bash
./setup-venv.sh
```

2. (Optional) Run `crawler.py` to fetch rating reviews from ESRB.
```bash
python3 crawler.py
```
You may modify the range in line 57 to download more data.
```python3
for i in range(5, 8):
```

Trying to crawl too many pages at once will result in timeout and indefinite hold.
I do not recommend fetching more than 4 pages at once.

3. Run `convert.py` to collect the fetched reviews into a csv file.
```bash
python3 convert.py
```
If you have modified `crawler.py` in step 2, you should modify the range in line 6 as well.
```python3
for i in range(1, 8):
```

4. Run `model.py` to run the model.
```bash
python3 convert.py
```
