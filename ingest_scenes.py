import os, json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Generate embeddings
from openai import AzureOpenAI

api_endpoint = "https://gpt4o-eastus-eliz.openai.azure.com/"
api_key = ""

client = AzureOpenAI(api_key=api_key,
                     api_version="2024-06-01",
                     azure_endpoint=api_endpoint)

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # deployment name
    )
    return response.data[0].embedding

# 读取guitantou.json文件
# json_file_path = 'actionSummary-P36-1-en.json'
# json_file_path = 'actionSummary-三轮车2-api0601.json'
json_file_path = 'actionSummary.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    documents = json.load(file)

# 生成向量并添加到文档
for doc in documents:
    summary_vector = generate_embedding(doc['summary'])
    actions_vector = generate_embedding(doc['actions'])
    character_vector = generate_embedding(doc['characters'])
    keyobjects_vector = generate_embedding(doc['key_objects'])
    keyactions_vector = generate_embedding(doc['key_actions'])
    nextaction_vector = generate_embedding(doc['next_action'])

    # Add vectors to the document
    doc['summaryVector'] = summary_vector
    doc['actionsVector'] = actions_vector
    doc['characterVector'] = character_vector
    doc['keyobjectsVector'] = keyobjects_vector
    doc['keyactionsVector'] = keyactions_vector
    doc['nextactionVector'] = nextaction_vector

# Save updated documents to a new JSON file with a "_with_vectors" suffix
base_name, ext = os.path.splitext(json_file_path)
new_file_path = base_name + "_with_vectors" + ext
print(f"Saving updated documents with vectors to: {new_file_path}")

# 保存更新后的文档
with open(new_file_path, 'w', encoding='utf-8') as file:
    json.dump(documents, file, ensure_ascii=False, indent=4)

# Start Ingesting...
# Define your service and index names
service_name = "cobra-video-search-eliz"
admin_key = ""

# index_name = "p36-1_index"  # cut in
index_name = "tricycle_index"  # cow, put english and chinese into one index together, same as line 78

# Create a SearchIndexClient to manage the index
index_client = SearchIndexClient(
    endpoint=f"https://{service_name}.search.windows.net",
    credential=AzureKeyCredential(admin_key),
)

# Define the schema of the index
"""
index_schema = {
    "name": "urban_scene_index",
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "sortable": True,
            "filterable": True,
        },
        {
            "name": "Start_Timestamp",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
            "facetable": True,
        },
        {
            "name": "End_Timestamp",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
            "facetable": True,
        },
        {
            "name": "sentiment",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
            "facetable": True,
        },
        {
            "name": "scene_theme",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
            "facetable": True,
        },
        {
            "name": "characters",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "facetable": False,
        },
        {
            "name": "summary",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "actions",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "key_objects",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
    ],
    "suggesters": [
        {
            "name": "sg",
            "searchMode": "analyzingInfixMatching",
            "sourceFields": [
                "scene_theme",
                "characters",
                "summary",
                "actions",
                "key_objects",
            ],
        }
    ],
    "scoringProfiles": [],
    "defaultScoringProfile": None,
    "corsOptions": {"allowedOrigins": ["*"], "maxAgeInSeconds": 300},
    "analyzers": [],
    "charFilters": [],
    "tokenFilters": [],
    "tokenizers": [],
    "semantic": {
        "configurations": [
            {
                "name": "urban_scene_semantic_config",
                "prioritizedFields": {
                    "titleField": {"fieldName": "summary"},
                    "prioritizedContentFields": [
                        {"fieldName": "summary"},
                        {"fieldName": "actions"},
                        {"fieldName": "characters"},
                    ],
                },
            }
        ]
    },
}

# Create or update the index
try:
    index = SearchIndex.deserialize(index_schema)
    index_client.create_or_update_index(index)
    print("Index created or updated successfully.")
except Exception as e:
    print(f"Failed to create or update index: {e}")
"""

# Create a SearchClient to upload documents
search_client = SearchClient(
    endpoint=f"https://{service_name}.search.windows.net",
    index_name=index_name,
    credential=AzureKeyCredential(admin_key),
)

# Load JSON data from the file
with open(new_file_path, "r") as json_file:
    data = json.load(json_file)


# Add IDs to documents
for i, doc in enumerate(data):
    doc["id"] = str(i)
    # Eliza added this code to convert key_objects to a string
    if isinstance(doc.get("key_objects"), list):
        doc["key_objects"] = ", ".join(doc["key_objects"])

# Validate documents against the expected schema
expected_fields = {
    "id": str,
    "Start_Timestamp": str,
    "End_Timestamp": str,
    "sentiment": str,
    "scene_theme": str,
    "characters": str,
    "summary": str,
    "actions": str,
    "key_objects": str,
    "summaryVector": list,
    "actionVector": list,
    "characterVector": list
}

# def validate_document(doc):
#     for field, field_type in expected_fields.items():
#         if field not in doc:
#             logging.error(f"validate document: Missing field '{field}' in document: {doc}")
#             return False
#         if not isinstance(doc[field], field_type):
#             logging.error(f"validate document: Field '{field}' has incorrect type in document: {doc}. Expected type: {field_type.__name__}, Actual type: {type(doc[field]).__name__}")
#             return False
#     return True

# valid_data = [doc for doc in data if validate_document(doc)]

# Upload documents to the index
try:
    result = search_client.upload_documents(documents=data)
    print(f"Documents uploaded successfully: {result}")
except Exception as e:
    print(f"Failed to upload documents: {e}")
