API_KEY = input("Enter your Gemini API key: ")
pinecone_api_key = input("Enter your pinecone API key: ")
chrome_extension_id= "enbghdihljfhpiepcnefpmkccbcdekok"

with open('.env', "w") as fp:
	fp.write(f"""API_KEY={API_KEY} 
pinecone_api_key={pinecone_api_key}
chrome_extension_id={chrome_extension_id}
""")

print("created enviornment file!")