from openai import OpenAI
import streamlit as st
import os
import getpass
import uuid


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

os.environ["OPENAI_API_KEY"] = ""

from flow_agent import FlowAgent
from flow_agent import MockToolResult

st.title("Flow Agent Chat")

# Mock tool result

# Add h1 header for mock tool values
st.header("Mock tool values")


# Add radio button for user_in_service_range
MockToolResult.user_in_service_range = st.radio(
    "Is user in service range?",
    [True, False],
    index=0,  # Default to True
    help="Toggle whether the user is in service range"
)

MockToolResult.check_knowledge_base_and_verify_service_result = st.radio(
    "Check knowledge base and verify we can service the user",
    [True, False],
    index=0,  # Default to True
)

MockToolResult.customer_distance_from_portland_result = st.selectbox(
    "Customer distance from Portland",
    [
        "Customer is more than 60 minutes away",
        "Customer is less than 45 minutes away", 
        "Customer is 45-60 minutes away"
    ],
)

MockToolResult.customer_need_booking_type = st.selectbox(
    "Customer Booking Type",
    [
        "ESTIMATE booking",
        "SERVICE CALL booking"
    ],
)

MockToolResult.upcoming_appointment_time = st.selectbox(
    "Upcoming Appointment Time",
    [
        "Today 10 AM",
        "10 AM Tomorrow",
        "10 PM Today"
    ],
)

MockToolResult.quoted_window_information = st.selectbox(
    "Quoted Window Information",
    [
        "Before quoted window",
        "During/After quoted window"
    ],
)

MockToolResult.booking_availability_for_emergency = st.radio(
    "Booking availability for emergency",
    [True, False],
    index=0,  # Default to True
    help="Toggle whether the booking is available for emergency"
)


if "agent" not in st.session_state:
    st.session_state["agent"] = FlowAgent("plumbing_flow_at.yaml")
    st.session_state["thread_id"] = str(uuid.uuid4())


if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Conversation:")
st.write("<br><br><br>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        agent = st.session_state["agent"]
        thread_id = st.session_state["thread_id"]
        response = agent.chat(user_query, thread_id=thread_id)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})