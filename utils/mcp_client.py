import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils.llm import MODEL

# which MCP server to launch - here: the filesystem server, scoped to ./data
# swap command/args to use the sqlite or github server instead

server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
)


async def ask(question):
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ask the server what tools it has, in Ollama's expected shape
            tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            } for t in (await session.list_tools()).tools]

            messages = [{"role": "user", "content": question}]
            res = ollama.chat(model=MODEL, messages=messages, tools=tools)

            # no tools needed -> just answer
            if not res.message.tool_calls:
                return res.message.content

            # run each tool the model picked on the MCP server, send results back
            messages.append(res.message)
            for call in res.message.tool_calls:
                result = await session.call_tool(
                    call.function.name,
                    call.function.arguments,
                )
                text = result.content[0].text if result.content else ""
                messages.append({
                    "role": "tool",
                    "content": text,
                    "tool_name": call.function.name,
                })

            # all tool results gathered -> one final answer
            final = ollama.chat(model=MODEL, messages=messages)
            return final.message.content
