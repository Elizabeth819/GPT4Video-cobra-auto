import { NextResponse } from "next/server";
import { NextApiResponse } from "next";
const { SearchClient, AzureKeyCredential } = require("@azure/search-documents");
require('dotenv').config();

export async function POST(req) {
  const client = new SearchClient(
    process.env.SEARCH_ENDPOINT,
    process.env.INDEX_NAME,
    new AzureKeyCredential(process.env.SEARCH_API_KEY)
  );
    // Extract the `messages` from the body of the request
    const { messages } = await req.json()
    // // Check if vectors are provided and valid
    // if (!vectors || !vectors.summaryVector || !vectors.actionsVector || !vectors.characterVector || !vectors.keyobjectsVector || !vectors.keyactionsVector || !vectors.nextactionVector) {
    //   return NextResponse.json({
    //     message: "Vectors data is missing or incomplete."
    //   }, {
    //     status: 400,
    //   });
    // }    
    // const searchResults = await client.search(messages,{queryType:"semantic",queryLanguage:"en-us",top:5,semanticConfiguration:"urban_scene_semantic_config"})
    // Elizabeth's code below parameters
    // Define the search parameters
    // Define the vector fields
    // const vectorFields = [
    //   { field: "summaryVector", vector: vectors.summaryVector },
    //   { field: "actionsVector", vector: vectors.actionsVector },
    //   { field: "characterVector", vector: vectors.characterVector },
    //   { field: "keyobjectsVector", vector: vectors.keyobjectsVector },
    //   { field: "keyactionsVector", vector: vectors.keyactionsVector },
    //   { field: "nextactionVector", vector: vectors.nextactionVector },
    // ];
    // const searchParams = {
    //   search: messages,
    //   queryType: "semantic",
    //   queryLanguage: "en-us",
    //   top: 5,
    //   semanticConfiguration: "urban_scene_semantic_config"
    //   // vector: {
    //   //   value: vectors.summaryVector, // This should be an array representing the vector
    //   //   fields: "summaryVector", // The name of the vector field in your index
    //   //   k: 5 // Number of top results to return based on vector similarity
    //   // }
    // };

    // Perform the search
    const searchResults = await client.search(messages,{queryType:"semantic",queryLanguage:"en-us",top:5,semanticConfiguration:"urban_scene_semantic_config"});
    
    let docs=[]
    for await (const result of searchResults.results) {
      console.log(result)
      docs.push(result.document)
    }
   return NextResponse.json({
    message: docs
  }, {
    status: 200,
  })
  }