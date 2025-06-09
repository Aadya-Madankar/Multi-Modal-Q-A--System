from langchain.tools import tool

@tool("Example PDF Processing Tool (Not Actively Used)")
def example_process_pdf_tool(pdf_path: str) -> str:
    print(f"CrewAI tool 'example_process_pdf_tool' called with {pdf_path}. (Not fully implemented for this simple version)")
    return f"Example tool call for {pdf_path}"

# In the next update the Crew AI framework is added.
# This part is just to show that we are on the way of building automous apllication. (Crew AI will be updated with multiple tools)
