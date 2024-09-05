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

api_endpoint = "https://aoai-eastus2-eliza.openai.azure.com/"
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
json_file_path = 'actionSummary.json'

with open(json_file_path, 'r', encoding='utf-8') as file:
    documents = json.load(file)

# 生成向量并添加到文档
for doc in documents:
    # Add vectors to the document
    doc['summaryVector'] = generate_embedding(doc['summary'])
    doc['actionsVector'] = generate_embedding(doc['actions'])
    doc['characterVector'] = generate_embedding(doc['characters'])
    doc['keyobjectsVector'] = generate_embedding(doc['key_objects'])
    doc['keyactionsVector'] = generate_embedding(doc['key_actions'])
    doc['nextactionVector'] = generate_embedding(doc['next_action'])

# Start Ingesting...
# Define your service and index names
service_name = "cobra-video-search-eliz"
admin_key = ""

# Define the name of the index
index_name = "complexscene2_index" 

# Create a SearchIndexClient to manage the index
index_client = SearchIndexClient(
    endpoint=f"https://{service_name}.search.windows.net",
    credential=AzureKeyCredential(admin_key),
)

# Define the schema of the index
index_schema = {
    "name": "complexscene2_index",
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
            "name": "characterVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
        },        
        {
            "name": "summary",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "summaryVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
        },        
        {
            "name": "actions",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "actionsVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
        },           
        {
            "name": "key_objects",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "keyobjectsVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
        },                     
        {
            "name": "key_actions",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "keyactionsVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
        },        
        {
            "name": "next_action",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "facetable": False,
        },
        {
            "name": "nextactionVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "facetable": False,
            "vectorSearchDimensions": 3072,
            "vectorSearchProfile": "vector-profile",            
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
                "name": "complex_scene_semantic_config",
                "prioritizedFields": {
                    "titleField": {"fieldName": "summary"},
                    "prioritizedContentFields": [
                        {"fieldName": "summary"},
                        {"fieldName": "actions"},
                        {"fieldName": "characters"},
                        {"fieldName": "key_objects"},
                        {"fieldName": "key_actions"},
                        {"fieldName": "next_action"},
                    ],
                },
            }
        ]
    },
  "vectorSearch": {
    "algorithms": [
      {
        "name": "vector-config-hnsw",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        },
        "exhaustiveKnnParameters": None
      }
    ],
    "profiles": [
      {
        "name": "vector-profile",
        "algorithm": "vector-config-hnsw",
        "vectorizer": "text-embedding-3-large",
      }
    ],
    "vectorizers": [
      {
        "name": "text-embedding-3-large",
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
          "resourceUri": "https://aoai-eastus2-eliza.openai.azure.com/",
          "deploymentId": "text-embedding-3-large",
          "apiKey": "44f7eecb1e684397b55156ba572ea23d",
          "modelName": "text-embedding-3-large",
          "authIdentity": None
        },
        "customWebApiParameters": None,
        "aiServicesVisionParameters": None,
        "amlParameters": None
      }
    ],
  }
}

# Create or update the index
try:
    index = SearchIndex.deserialize(index_schema)
    index_client.create_or_update_index(index)
    print("Index created or updated successfully.")
except Exception as e:
    print(f"Failed to create or update index: {e}")


# Create a SearchClient to upload documents
search_client = SearchClient(
    endpoint=f"https://{service_name}.search.windows.net",
    index_name=index_name,
    credential=AzureKeyCredential(admin_key),
)

# Add IDs to documents
for i, doc in enumerate(documents):
    doc["id"] = str(i)
  
# 构建索引操作
upload_documents = [
    {
        "@search.action": "upload",  # 或者 "mergeOrUpload" 或 "delete"
        "id": doc["id"],
        "Start_Timestamp": doc.get("Start_Timestamp", ""),
        "End_Timestamp": doc.get("End_Timestamp", ""),
        "sentiment": doc.get("sentiment", ""),
        "scene_theme": doc.get("scene_theme", ""),
        "characters": doc.get("characters", ""),
        "summary": doc.get("summary", ""),
        "actions": doc.get("actions", ""),
        "key_objects": doc.get("key_objects", ""),
        "key_actions": doc.get("key_actions", ""),
        "next_action": doc.get("next_action", ""),
        "summaryVector": doc.get("summaryVector", []),
        "actionsVector": doc.get("actionsVector", []),
        "characterVector": doc.get("characterVector", []),
        "keyobjectsVector": doc.get("keyobjectsVector", []),
        "keyactionsVector": doc.get("keyactionsVector", []),
        "nextactionVector": doc.get("nextactionVector", []),
    }
    for doc in documents
]

# Upload documents to the index
try:
    result = search_client.upload_documents(documents=upload_documents)
    if result[0].succeeded:
        print(f"Documents uploaded successfully: {result}")
    else:
        logging.error(f"Failed to upload documents: {result[0].error_message}")    
except Exception as e:
    print(f"Failed to upload documents: {e}")
