from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import json

# Initialize the model
model = ChatOllama(model="llama3.1")

# Define the request model
class ConversionRequest(BaseModel):
    amount: float
    base_currency: str
    target_currency: str

# Create tools
@tool
def get_conversion(base_currency: str, target_currency: str) -> dict:
    """Fetches the currency conversion rate."""
    url = f'https://v6.exchangerate-api.com/v6/fae573cd0eff3ed235f7de34/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()

@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Converts the base currency value to the target currency."""
    return base_currency_value * conversion_rate

# Initialize the agent with tools
agent = model.bind_tools([get_conversion, convert])

# Initialize FastAPI app
app = FastAPI()

@app.post("/convert")
async def convert_currency(request: ConversionRequest):
    try:
        # Prepare the user query
        user_query = f"Convert {request.amount} {request.base_currency} to {request.target_currency}."
        messages = [HumanMessage(content=user_query)]

        # Invoke the model
        ai_response = agent.invoke(messages)

        # Initialize variables
        conversion_rate = None
        converted_amount = None
        assistant_text = ""

        for tool_call in ai_response.tool_calls:
            if tool_call["name"] == "get_conversion":
                tool_message = get_conversion.invoke(tool_call)
                if isinstance(tool_message.content, str):
                    conversion_data = json.loads(tool_message.content)
                else:
                    conversion_data = tool_message.content

                conversion_rate = conversion_data.get("conversion_rate")
                assistant_text += f"Rate: {conversion_rate}\n"
                messages.append(tool_message)

            elif tool_call["name"] == "convert" and conversion_rate is not None:
                tool_call["args"]["conversion_rate"] = conversion_rate
                tool_message = convert.invoke(tool_call)
                converted_amount = tool_message
                assistant_text += f"Result: {converted_amount}\n"
                messages.append(tool_message)

        if converted_amount is not None:
            return {"converted_amount": converted_amount}
        else:
            raise HTTPException(status_code=400, detail="Conversion failed.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"External service error: {e}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing expected data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
