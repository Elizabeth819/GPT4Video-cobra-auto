Hi everyone, general information and format for each of the .env file is below. (backend and frontend)

## Important Notes
  - Use GPT-4o instead of GPT-4-turbo vision for latest video interpretation capability.
  - The Azure GPT4 Vision service has 2 issues, 1: you can only send 10 (now 20, but unstable) images per call, so max FPI is 10, and you need to apply to turn of content filtering, as it is synchronous and adds 30+ seconds to each call.

## Frameworks and Languages:
  - Python 3.12 for the backend
  - NodeJS 18 and greater for the frontend, uses NextJS

## Installation Instructions:

1. **Clone the Repository**:
   - Use `git clone` to clone the project from GitHub.

2. **Setup the environment**:
   - **For Mac**:
     - Install **CMake** and other dependencies beforehand:
       ```
       brew install ffmpeg
       brew install cmake
       ```
   - **For Linux (Ubuntu)**:
     - Install **CMake**:
       ```
       sudo apt-get install cmake libgl1
       ```
   - **For Windows with VS Code**:
     - Open the folder using **devcontainer** in VS Code, and it will automatically install all the necessary backend dependencies via `installdependencies.sh`.

3. **Backend Setup**:
   - Navigate to the backend directory and install the required Python packages:
     ```
     pip install -r requirements.txt
     ```

4. **Frontend Setup**:
   - Navigate to the **COBRA** folder and install the frontend dependencies:
     ```
     cd COBRA
     sudo apt install npm
     npm install
     npm install next
     ```
5. **Set the Environment Variables (ENV)**:
  - Env Template Backend (.envsample copy to .env)
    - AZURE_SPEECH_KEY= "This is not neccesary, it is for an eventual azure speech integration"
    - AZURE_WHISPER_KEY= "This is the key to your whisper instance in azure"
    - AZURE_WHISPER_DEPLOYMENT="deployment name of you azure whisper"
    - AZURE_WHISPER_ENDPOINT="Full endpoint url of your azure whisper 'https://jfjafjajf'"
    - AZURE_VISION_KEY="Your Azure OpenAI key for GPT4V"
    - AUDIO_API_TYPE="Azure or OpenAI"
    - VISION_API_TYPE="Azure or OpenAI"
    - OPENAI_API_KEY="OpenAI API key"
    - VISION_DEPLOYMENT_NAME="Azure OpenAI GPT4V deployment name"
    - VISION_ENDPOINT="Just endpoint name for Azure GPT4V 'ovv-vision'"

  - Env Template Frontend (./COBRA/.envsample copy to ./COBRA/.env)
    - AZ_OPENAI_KEY="Azure OpenAI key"
    - AZ_OPENAI_REG=canadaeast
    - AZ_OPENAI_BASE=https://{your endpoint name}.openai.azure.com/
    - AZ_OPENAI_VERSION=2024-02-01
    - GPT4=4turbo
    - SEARCH_ENDPOINT=https://{your_search_service).search.windows.net
    - SEARCH_API_KEY="Your Search API Key"
    - INDEX_NAME=Your Index Name

## Run Instructions Backend / 後端執行說明:

- Check if frames and audio folder exist, if existing, delete them, they are intermediate processing frames and extracted audios during running below ActionSummary-predict_explain-fsl-sys-Cn-cutin.py or ActionSummary.py. <br>
  檢查frames和audio資料夾是否存在，如果存在，請刪除它們。這些是執行下列的 ActionSummary-predict_explain-fsl-sys-Cn-cutin.py 或 ActionSummary.py 時的中間處理過程中生成的影像幀和提取的音訊。
- For autonomous driving video analysis 針對自動駕駛視頻分析
  run the ActionSummary-predict_explain-fsl-sys-Cn-cutin.py, video file name can be changed in the last line before main entry at the bottom. <br>
  執行 ActionSummary-predict_explain-fsl-sys-Cn-cutin.py，可在腳本底部的main入口前更改視頻文件名稱。
- For general video analysis 針對一般視頻分析
  - To execute the **ActionSummary.py** script from the terminal, use the following command format:
    ```
    python ActionSummary.py "path_to_video" "frame_interval" "frames_per_interval" "False"
    ```
  - Example:
    ```
    $ python ActionSummary.py "./test_video/mcar-driving.mov" 10 10 False
    ```
    In this example:
    - `"./myVideo.mp4"` is the path to the video file.
    - `10` is the frame interval (the number of frames to skip between captures).
    - `10` is the number of frames to process per interval.
    - `False` is a boolean value (in this case, indicating not to enable certain additional options).
- For ChapterAnalysis 針對章節分析
  in the code, change the video path and fi/fpi directly at the end of the script. <br>
  在程式碼中，直接在腳本末尾更改視頻路徑和fi/fpi

## Run Instructions Frontend:
1. **Install necessary packages**:
   - For **Linux shell**, run the following commands to install npm and required packages:
     ```
     sudo apt install npm
     npm install
     npm install next
     ```
2. **Start the development server**:
   - Navigate to the **COBRA** folder and run:
     ```
     npm run dev
     ```
3. **Fixing Next.js errors**:
   - If you encounter a **Next.js error**, delete the `.next` folder from the project directory and try restarting the server.

4. **Running in production**:
   - To run the project in production mode, first build the project:
     ```
     npm run build
     ```
   - Then, start the server with:
     ```
     npm start
     ```
   - Alternatively, you can use:
     ```
     npm run dev
     ```

5. **Azure VM configuration**:
   - If using an **Azure Virtual Machine (VM)**, add an inbound policy:
     - Navigate to **Settings > Network** and add a policy for port **3000** with the following:
       - **Destination port**: 3000
       - **Protocol**: TCP
       - **Action**: Allow
       - **Priority**: 100

6. **Modifications in player.jsx**:
   - In **COBRA/components/component/player.jsx**, make the following changes:
     1. At **line 15**, change the name of the **actionSummary.json** file. Copy the updated JSON file to the **COBRA/app/data** folder.
     2. At **line 317**, change the video file name in the `SetVideoURL` function. Copy the video file to the **COBRA/public** folder. The frontend will read the video from here.

7. **Accessing the application**:
   - Open the application in your local web browser at:
     ```
     localhost:3000
     ```
   - Or, if connecting to a remote VM, use:
     ```
     your-vm-public-ip:3000
     ``` 

## 在前端中產生資料, 並提供chatbot功能：
  - 首先執行 `ActionSummary.py` 生成 `actionSummary.json` 文件：
    - 該文件包含視頻中的所有動作及其對應的時間戳。
    - 將此文件手動移至 `./COBRA/app/data/` 下，這樣 UI 上就會顯示資料。
  - 執行 `ingest_scenes.py` 來創建索引並將 `actionSummary.json` 整合到 AI 搜索中。

### `ingest_scenes.py` 中的配置

- 修改以下 Azure OpenAI 服務和 API 金鑰：

```python
api_endpoint = "https://aoai-XXXXX.openai.azure.com/"
api_key = "XXXXX"
```

- 並修改你的嵌入模型名稱：

```python
def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # 部署名稱
    )
```

- 定義你的Azure AI Search service name和索引名稱：

```python
service_name = "cobra-video-search-eliz"
admin_key = ""
index_name = "complexscene2_index"
```

- 修改 `actionSummary.json` 文件的路徑：

```python
json_file_path = 'actionSummary.json'
```

  - 搜索服務必須至少為 "**Standard**" 價格層級，否則無法使用 "語義搜索" 功能。
  - 修改 `app/api/cog/route.js` 文件中的最後幾行，將語義配置名稱改為與 `ingest_scenes.py` 中索引創建時使用的語義配置相同。
  - 在 Cobra 環境文件中修改索引名稱，使其與 `ingest_scenes.py` 保持一致。
  - 在搜尋框中使用語義方式搜索，即可獲得語義相似的結果，而不需要與關鍵字完全匹配。

## 前端視頻顯示
- 在 `Cobra/components/component/player.jsx` 中：
  - 將第 320 行左右的視頻文件名更改為你的視頻文件名：

```javascript
const [videoURL, setVideoURL] = useState('./car-driving.mov')
```

  - 你的視頻文件應該位於 `cobra/public/car-driving.mov` 路徑下。

### 在 `COBRA/ingest.py` 中

- 替換為你的 Azure Search 服務、API 金鑰、索引名稱和 JSON 文件路徑：

```python
search_service_name = "video-cobra"
admin_api_key = "XXXXX"
index_name = "c-index"
json_file_path = "app/data/chapterBreakdown_ingest.json"
```

## Containers:
  - There are Docker Containers ready to go for both frontend and backend, as well as a compose that allows for facial recognition.

## 增強功能：
  - 使用 Florence 預處理幀作為嵌入，以便我們能夠最佳化地為任何給定的 FI 決定最佳 FPI，使解決方案在深度上始終保持最佳化。
  - 全面的 UI 體驗，具有上傳和處理視頻的功能。
  - 更好的 Azure Whisper/語音支持。
  - 正確的音頻時間戳和說話人分辨。
  - 更完善的 ENV 策略。
  - 當不使用預覽時，與 response_format 的整合。

## 優先使用案例：
  - 連續動作檢測、解釋和預測。
  - 關鍵動作和廣泛對象的識別。
  - 為視障人士進行音頻配音，準確插入文本和神經語音。
  - 通過語義插入進行動態廣告插播。
  - 多視頻集的分析。
  - 片尾字幕 ID。
  - 角色跟踪/識別。
  - 視頻內容審核。
  - 異常檢測。
  - 情感情緒分析。
