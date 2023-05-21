# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
#set -e

echo "About to get readthedocs"

wget -r -A.html https://python.langchain.com/en/latest/
python3 ingest.py
