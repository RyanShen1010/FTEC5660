# Download full task3.json from UNBench
curl -sL "https://raw.githubusercontent.com/yueqingliang1/UNBench/main/data/task3.json" -o data/task3.json
echo "Downloaded. Samples: $(python -c "import json; d=json.load(open('data/task3.json')); print(len(d['drafts']))")"
