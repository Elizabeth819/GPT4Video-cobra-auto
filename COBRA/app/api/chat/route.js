import { OpenAIStream, StreamingTextResponse } from 'ai'
const { OpenAIClient, AzureKeyCredential } = require("@azure/openai");
const { SearchClient } = require("@azure/search-documents");
require('dotenv').config()





// Create an OpenAI API client (that's edge friendly!)
function streamChatCompletions(client, deploymentId, messages, options) {
    const events = client.listChatCompletions(deploymentId, messages, options);
    //console.log(events)
    const stream = new ReadableStream({
      async start(controller) {
        for await (const event of events) {
          controller.enqueue(event);
        }
        controller.close();
      },
    });
  
    return stream;
  }
 
export async function POST(req) {
  // Extract the `messages` from the body of the request
  const { messages } = await req.json()
 
  // Ask OpenAI for a streaming chat completion given the prompt
  const client = new OpenAIClient(
    process.env.AZ_OPENAI_BASE,
  new AzureKeyCredential(process.env.AZ_OPENAI_KEY));
  const client2 = new SearchClient(
    process.env.SEARCH_ENDPOINT,
    process.env.INDEX_NAME,
    new AzureKeyCredential(process.env.SEARCH_API_KEY)
  );

    // Extract the `messages` from the body of the request
    
const searchResults = await client2.search(messages[messages.length-1].content, {queryType: "simple", queryLanguage: "en-us", top: 5});

    let docs=[]
    for await (const result of searchResults.results) {
      docs.push(result.document["content"]+"\n Source:"+result.document["source"])
    }
 
    messages[0].content=messages[0].content+"\n CONTEXT FROM SOURCE:\n"+JSON.stringify(docs)
    
    //console.log(JSON.stringify(messages))
  const stream = await streamChatCompletions(client,process.env.GPT4,messages)
    

  // Convert the response into a friendly text-stream
  const streams = OpenAIStream(stream)
  console.log("sending")
  // Respond with the stream
  return new StreamingTextResponse(streams)
}