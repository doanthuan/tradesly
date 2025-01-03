import yaml
import re
from langchain_core.tools import tool
import random
from datetime import datetime
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from textwrap import dedent
from langchain_core.messages.tool import ToolMessage
from langfuse.decorators import observe
from langchain_openai import ChatOpenAI


from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-7a32fef8-8e0b-41b0-8d93-147a18cc40fd",
    public_key="pk-lf-c05ba234-0e69-44dd-bbba-f2e7b6f4b61f",
    host="https://us.cloud.langfuse.com", # 🇺🇸 US region
)


class Flow(object):

    def __init__(self, flow_file):
        '''
        Construct the flow based on the descriptions from flow_file

        Args:
            flow_file (str): the path pointed to the text file containing the flow instructions.
        '''

        self.plan_data = None
        with open(flow_file) as stream:
            try:
                self.plan_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.current_step = 0

    def get_current_task(self):
        return self.plan_data["step_" + str(self.current_step)]

    def parse_step_number(self, text: str):
        if "step" in text:
            match = re.search(r'step_(\d+)', text)
            if not match:
                match = re.search(r'step (\d+)', text)
            if match:
                number = match.group(1)
                return int(number.strip())
            
        return None
    
    def go_next(self):
        self.current_step += 1
        return self.get_current_task()

    def go_to(self, step_number):
        self.current_step = step_number
        return self.get_current_task()
    
    def reset(self):
        self.current_step = 0
        return self.get_current_task()


class MockToolResult:
    user_in_service_range = False
    check_knowledge_base_and_verify_service_result = False
    customer_distance_from_portland_result = "Customer is more than 60 minutes away"
    # customer_distance_from_portland_result = "Customer is less than 45 minutes away"
    # customer_distance_from_portland_result = "Customer is 45-60 minutes away"
    customer_need_booking_type = "ESTIMATE booking"
    # customer_need_booking_type = "SERVICE CALL booking"
    upcoming_appointment_time = "Today 10 AM"
    quoted_window_information = "During quoted window"
    booking_availability_for_emergency = True


@tool
def go_to_next_step(step_number: int) -> str:
    """Use this tool to guide user to go to next step"""

    return "Have gone to step " + str(step_number)


@tool
def log_customer_info(name: str, address: str, phone_number: str, email_address: str) -> str:
    """Log customer info to Talkdesk Contact"""
    
    print(f"Logging customer info: {name}, {address}, {phone_number}, {email_address}")

    return "Customer info logged"

@tool
def verify_customer_in_service_range() -> str:
    """Verify customer is in service range using customer Map"""

    if MockToolResult.user_in_service_range:
        return "Customer is verified to be in service range"
    else:
        return "Customer is verified to be out of service range"

@tool
def check_knowledge_base():
    """Tool to check knowledge base list of provided services"""

    if MockToolResult.check_knowledge_base_and_verify_service_result:
        return "Have checked the knowledge base and verified that we can service the customer"
    else:
        return "Have checked the knowledge base and verified that we cannot service the customer"


@tool
def confirm_customer_distance_from_portland_me():
    """Service Zone Lookup tool. Use this tool to check customer's distance from Portland, ME"""

    return f"{MockToolResult.customer_distance_from_portland_result}. Customer booking type: {MockToolResult.customer_need_booking_type}"

@tool
def schedule_next_available_appointment():
    """Schedule next available estimate appointment. Only use this tool when your current task is to schedule next available estimate appointment."""

    return "Next available appointment scheduled"


@tool
def get_upcoming_appointment_time():
    """Get upcoming appointment time. Only use this tool if customer want to change their upcoming appointment time."""

    return f"Have received upcoming appointment time: {MockToolResult.upcoming_appointment_time}"


@tool
def send_sms_message_to_user():
    """Send SMS message to user"""

    return "SMS message sent to user"


@tool
def get_quoted_window_info():
    """Get quoted window information"""

    return MockToolResult.quoted_window_information

@tool
def cancel_upcoming_appointment():
    """Cancel upcoming appointment"""

    return "Upcoming appointment cancelled"

@tool
def check_booking_availability_for_emergency():
    """Use this tool to check booking availability for emergency"""

    if MockToolResult.booking_availability_for_emergency:
        return "Available booking for emergency found."
    else:
        return "No available booking for emergency found."


current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

demo_tools = [
    go_to_next_step,
    log_customer_info,
    verify_customer_in_service_range,
    check_knowledge_base,
    confirm_customer_distance_from_portland_me,
    schedule_next_available_appointment,
    get_upcoming_appointment_time,
    send_sms_message_to_user,
    get_quoted_window_info,
    cancel_upcoming_appointment,
    check_booking_availability_for_emergency
]


def build_agent():

    llm = ChatOpenAI(model="gpt-4o-mini")

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""
                       
    **Context:**
                       
    You are agent for Plumbing Service handling inbound calls.
    Currently, you are in step number: {current_step} of a workflow process.
    Your task is to follow task instructions, determine the condition and go to the next step.

    **Current Task:**
    {current_task}

    **Last tool message:**
    {last_tool_message}

    **Current time:**
    {datetime}
                       
    **Notes:**
    - If you need to determine the condition for the next step, read your conversation history as well as last tool message to make a decision. If you are still not sure, ask user to clarify.
    - Go to next step by calling go_to_next_step tool.
    - If you have some example questions in your task instructions, please ask user one by one until you clear the condition.
                    
    **Tools usage:**
    - You're provided with tools that you can select to implement your task steps. Choose the best matching tool for a step.
    - ALWAYS call 1 tool at a time.
    

                """)
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(datetime=current_datetime)

    assistant_runnable = primary_assistant_prompt | llm.bind_tools(demo_tools)
    return assistant_runnable


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_task: str
    current_step: int
    last_tool_message: str


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def last_tool_message(state, messages_key: str = "messages") -> str:
    if isinstance(state, list):
        last_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        last_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        last_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if isinstance(last_message, ToolMessage):
        return last_message.content
    else:
        return ""
    

def update_state(state: State, flow: Flow):
    last_message = last_tool_message(state)
    if last_message == "":
        return None
    
    if "Have gone to step " in last_message:
        step_number = flow.parse_step_number(last_message)
        if not step_number:
            raise ValueError("Invalid step number:", last_message)    
        print("FLOW CHANGE: Go to step ", step_number)
        flow.go_to(step_number)
    else:
        return last_message
    

class Assistant:
    def __init__(self, runnable: Runnable, flow: Flow):
        self.runnable = runnable
        self.flow = flow

    @observe()
    def __call__(self, state: State, config: RunnableConfig):

        # verify
        
        while True:
            # configuration = config.get("configurable", {})
            # passenger_id = configuration.get("passenger_id", None)
            # state = {**state, "user_info": passenger_id}
            last_message = update_state(state, self.flow)
            current_task = self.flow.get_current_task()
            if "last_tool_message" not in state:
                state["last_tool_message"] = ""
            state = {
                        **state, 
                        "current_task": current_task, 
                        "current_step": self.flow.current_step,
                    }
            if last_message and len(last_message) > 0:
                print("UPDATING last_tool_message:", last_message)
                state["last_tool_message"] = last_message
            
            print("\n\nCURRENT TASK:", current_task)
            result = self.runnable.invoke(state, config={**config, "callbacks": [langfuse_handler]})
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                print("LLM returns empty response")
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        # Tool call or end of the flow
        return {"messages": result, "last_tool_message": last_message}


class FlowAgent:
    def __init__(self, flow_file: str):
        self.flow = Flow(flow_file)
        self.graph = FlowAgent.build_graph(self.flow)
    
    @classmethod
    def build_graph(cls, flow):

        assistant_runnable = build_agent()

        builder = StateGraph(State)

        # Define nodes: these do the work
        builder.add_node("assistant", Assistant(assistant_runnable, flow))
        builder.add_node("tools", create_tool_node_with_fallback(demo_tools))
        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        # The checkpointer lets the graph persist its state
        # this is a complete memory for the entire graph.
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        return graph

    def run(self, user_questions: list, thread_id: str):
        # Update with the backup file so we can restart from the original place in each section

        config = {
            "configurable": {
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            }
        }

        self.flow.reset()

        _printed = set()
        for question in user_questions:
            events = self.graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)




# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass


