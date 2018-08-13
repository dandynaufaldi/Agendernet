# Data directory

**Place the datasets here**

## Folder naming and content
- Save dataset in separate folder, preserve their original directory structure
- For Adience dataset, provide 2 folder, each for images and fold.txt file
- Suggested folder structure
  ```bash
  data/
    ├── adience
    ├── adience_fold
    ├── fgnet
    ├── imdb
    ├── utkface
    └── wiki
  ```

## Generating .csv db file
Use make_db.py
```bash
usage: make_db.py [-h] --db_name {imdb,wiki,utkface,fgnet,adience} --path PATH

optional arguments:
  -h, --help            show this help message and exit
  --db_name {imdb,wiki,utkface,fgnet,adience}
                        Dataset name
  --path PATH           Path to dataset folder
```
for Adience, use path to folder contain Adience fold.txt files