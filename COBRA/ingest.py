import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    ComplexField,
    SimpleField,
    _edm
)

# Replace with your Azure Search service and API key
search_service_name = "video-cobra"
admin_api_key = "Ja9lbzRSeohAQvBaJlfid6vtJdqBYittIzMGhYyKrrAzSeCAEZlt"
index_name = "c-index"
json_file_path = "app/data/chapterBreakdown_ingest.json"  # Replace with your local JSON file path

# Initialize the SearchIndexClient
index_client = SearchIndexClient(
    endpoint=f"https://{search_service_name}.search.windows.net",
    credential=AzureKeyCredential(admin_api_key)
)

# Define the index schema
# index_schema = SearchIndex(
#     name=index_name,
#     fields=[
#         SimpleField(name="chapter_id", type="Edm.String", key=True, searchable=False),
#         SimpleField(name="title", type=edm.String, searchable=True, filterable=True, sortable=True, facetable=True),
#         SimpleField(name="start_frame", type=edm.String, searchable=False, filterable=True, sortable=True, facetable=False),
#         SimpleField(name="end_frame", type=edm.String, searchable=False, filterable=True, sortable=True, facetable=False),
#         ComplexField(name="scenes", fields=[
#             SimpleField(name="title", type=edm.String, searchable=True, filterable=True, sortable=True, facetable=True),
#             SimpleField(name="description", type=edm.String, searchable=True, filterable=True, sortable=False, facetable=False)
#         ])
#     ]
# )

# Create the index
# index_client.create_index(index_schema)

# Initialize the SearchClient
search_client = SearchClient(
    endpoint=f"https://{search_service_name}.search.windows.net",
    index_name=index_name,
    credential=AzureKeyCredential(admin_api_key)
)

# Load JSON data from the file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Ingest the data into the index
search_client.upload_documents(documents=data)
print("Data uploaded successfully.")
