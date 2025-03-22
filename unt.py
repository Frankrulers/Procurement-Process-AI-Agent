#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# ## Authors- 
# ##### *Abhinav Kumar*- FT251002
# 
# ##### *Aditi Sharma*- FT251005
# 
# ##### *Aditya Gupta*- FT251006
# 
# ##### *Ambika Prasad Swain*- FT251013
# 
# ##### *Kaustuv Bhattacharya*- FT252043

# # Objective:
# ### The objective of this assignment is to design, implement, and evaluate an AI-driven procurement automation system for TransGlobal Industries. The solution should leverage Large Language Models (LLMs), LangChain, and Streamlit to streamline key procurement processes.
# 
# ### The overarching goals are- Reducing manual labor, increasing accuracy, removing bias, and speeding up procurement decision-making.

# ## Importing Necessary Libraries 

# In[315]:


import os
import streamlit as st
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from io import StringIO


# ## Checking Installed Package Versions in Python 

# In[318]:


import pkg_resources

openai_version = pkg_resources.get_distribution("openai").version
streamlit_version = pkg_resources.get_distribution("streamlit").version

try:
    import google.generativeai as genai
    print(f"google-generativeai version: {genai.__version__}")
except ImportError:
    print("google-generativeai not installed or version not accessible")

print(f"streamlit version: {st.__version__}")
print(f"langchain version: {langchain.__version__}")
print(f"pandas version: {pd.__version__}")


# ## 1. Initial Setup & Configurations

# In[228]:


# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set the GEMINI_API_KEY environment variable.")
    st.stop()


# In[229]:


# Initialize LLM with a low temperature (0.1)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)


# ## 2. Defining Prompt Templates and Chains for Each Step

# #### Step 1: Business to Technical Requirements Conversion

# In[232]:


tech_req_prompt = PromptTemplate(
    input_variables=["business_req"],
    template="""
    
Context: TransGlobal Industries is automating its procurement proces using AI Agents.
Role: You are a technical requirements analyst.
Task: Convert the following Business Requirements into a detailed Technical Requirements Document.
Action: Generate a structured Technical Requirements Document. Provide the output in plain text, without any Markdown formatting.

Business Requirement:
{business_req}

    The technical requirements document should include:
    
    1. A header with project title and date
    2. Numbered sections for different requirement categories
    3. For each requirement, include:
       * The specific technical requirement
    
    Each requirement should be specific, measurable, achievable, relevant, and time-bound (SMART).
    
    Ensure all functional and non-functional requirements are covered, including:
    - System architecture
    - Performance specifications
    - Integration requirements
    - Security requirements
    - User interface specifications
    - Data management requirements
    - Any relevant standards or compliance needs
    Ensure that you don't pick up LLM, streamlit, Langchain or any other component required to build the agent. The agent should be strictly based on the business required document.
"""
)
tech_req_chain = LLMChain(llm=llm, prompt=tech_req_prompt)


# #### Step 2: Vendor Shortlisting (LLM is used to get names of Vendors)

# In[234]:


vendor_shortlist_prompt = PromptTemplate(
    input_variables=["tech_req", "vendor_history"],
    template="""
Context: TransGlobal Industries is automating its procurement process.
Role: You are a vendor identification specialist.
Task: Based on the Technical Requirements and Vendor History, identify suitable vendors.
Action: Return a comma-separated list of vendors that might be suitable (e.g., Vendor A, Vendor B). Only list the vendor names, nothing else. Provide the output in plain text, without any Markdown formatting. 

Technical Requirements:
{tech_req}

Vendor History:
{vendor_history}
"""
)
vendor_shortlist_chain = LLMChain(llm=llm, prompt=vendor_shortlist_prompt)


# #### Step 3: Tender Document & RFP Preparation

# In[236]:


tender_doc_prompt = PromptTemplate(
    input_variables=["tech_req", "business_req"],
    template="""
Context: TransGlobal Industries is automating its procurement process.
Role: You are a procurement document specialist.
Task: Prepare a comprehensive Tender Document and Request for Proposal (RFP).
Action: Format the document to encapsulate the provided technical and business requirements.Provide the output in plain text, without any Markdown formatting.

Technical Requirements:
{tech_req}

Business Requirements:
{business_req}
"""
)
tender_doc_chain = LLMChain(llm=llm, prompt=tender_doc_prompt)


# #### Step 4: Tender Email Generation for Shortlisted Vendors

# In[238]:


tender_email_prompt = PromptTemplate(
    input_variables=["shortlisted_vendors", "tender_doc"],
    template="""
Context: TransGlobal Industries procurement process.
Role: You are a communications specialist.
Task: Generate a professional email to send the tender document to the shortlisted vendors.
Action: Provide the output in plain text, without any Markdown formatting.
The email should:
    1. Have a clear, professional subject line
    2. Introduce the company (TransGlobal Industries) and the opportunity briefly
    3. Mention that they've been shortlisted based on their capabilities
    4. Explain that the tender document is attached
    5. Specify a deadline for submission (3 weeks from now)
    6. Provide contact information for questions
    7. End with a professional closing
    
    Keep the email concise but professional. Do not include the actual tender document text in the email.
    Format as a complete email with Subject line, Greeting, Body, and Signature.

    The emails should be completely separate for each 2 selected vendor.

Shortlisted Vendors:
{shortlisted_vendors}

Tender Document:
{tender_doc}
"""
)
tender_email_chain = LLMChain(llm=llm, prompt=tender_email_prompt)


# #### Step 5: Bid Evaluation

# In[240]:


bid_evaluation_prompt = PromptTemplate(
    input_variables=["bids_data"],
    template="""
Context: TransGlobal Industries procurement process.
Role: You are a bid evaluation expert.
Task: Evaluate the provided bids based on price, quality, delivery, and technological capability.
Action: Identify and list the top two bids with scoring and brief justification. Provide the output in plain text, without any Markdown formatting.

Bids Data:
{bids_data}
"""
)
bid_evaluation_chain = LLMChain(llm=llm, prompt=bid_evaluation_prompt)


# #### Step 6: Negotiation Strategy and BATNA Analysis

# In[242]:


negotiation_strategy_prompt = PromptTemplate(
    input_variables=["top_two_bids"],
    template="""
Context: TransGlobal Industries procurement process.
Role: You are a negotiation strategist.
Task: Develop a negotiation strategy and identify the Best Alternative to a Negotiated Agreement (BATNA).
Action: Provide clear strategies and recommendations based on the top two bids. Provide the output in plain text, without any Markdown formatting.

Top Two Bids:
{top_two_bids}
"""
)
negotiation_strategy_chain = LLMChain(llm=llm, prompt=negotiation_strategy_prompt)


# #### Step 7: Risk Assessment Report Generation

# In[244]:


risk_assessment_prompt = PromptTemplate(
    input_variables=["negotiation_strategy", "bid_data"],
    template="""
Context: TransGlobal Industries procurement process.
Role: You are a risk assessment specialist.
Task: Generate a risk assessment report for the preferred vendor.
Action: Include analysis on delivery, quality, compliance, performance, and communication risks. Provide the output in plain text, without any Markdown formatting.

Negotiation Strategy:
{negotiation_strategy}

Bid Data:
{bid_data}
"""
)
risk_assessment_chain = LLMChain(llm=llm, prompt=risk_assessment_prompt)


# #### Step 8: Contract Document Generation

# In[246]:


contract_doc_prompt = PromptTemplate(
    input_variables=["risk_assessment"],
    template="""
Context: TransGlobal Industries procurement process.
Role: You are a contract drafting expert.
Task: Draft a comprehensive contract document.
Action: Include clauses on risk mitigation, performance guarantees, and dispute resolution based on the risk assessment report. Provide the output in plain text, without any Markdown formatting.

Risk Assessment Report:
{risk_assessment}
"""
)
contract_doc_chain = LLMChain(llm=llm, prompt=contract_doc_prompt)


# ## 3. Initializing Session State

# In[248]:


if "tech_req_doc" not in st.session_state:
    st.session_state.tech_req_doc = ""
if "shortlisted_vendors" not in st.session_state:
    st.session_state.shortlisted_vendors = ""
if "tender_doc" not in st.session_state:
    st.session_state.tender_doc = ""
if "tender_email" not in st.session_state:
    st.session_state.tender_email = ""
if "bid_evaluation" not in st.session_state:
    st.session_state.bid_evaluation = ""
if "negotiation_strategy" not in st.session_state:
    st.session_state.negotiation_strategy = ""
if "risk_assessment" not in st.session_state:
    st.session_state.risk_assessment = ""
if "contract_doc" not in st.session_state:
    st.session_state.contract_doc = ""
if "top_two_bids" not in st.session_state:
    st.session_state.top_two_bids = ""
if "vendor_history_df" not in st.session_state:
    st.session_state.vendor_history_df = None


# ## 4. Building the Streamlit UI

# In[250]:


# Configure page
st.set_page_config(page_title="TransGlobal Industries Procurement Automation System", layout="wide")

# Header with blue rectangle and white text
st.markdown(
    '<div style="background-color: blue; padding: 10px;">'
    '<h1 style="color: white; text-align: center;">TransGlobal Industries Procurement Automation System</h1>'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar listing procurement steps
st.sidebar.title("Procurement Process Steps")
steps = [
    "1. Business to Technical Requirements",
    "2. Vendor Shortlisting",
    "3. Tender & RFP Preparation",
    "4. Tender Email Generation",
    "5. Bid Evaluation",
    "6. Negotiation Strategy & BATNA",
    "7. Risk Assessment",
    "8. Contract Generation"
]
for step in steps:
    st.sidebar.write(step)


# ## 5. File Uploads & Inputs

# In[252]:


st.header("Input Section - Follow the sequence for uploading the files")


# #### Business Requirements Upload (Step 1)

# In[254]:


st.subheader("Upload Business Requirements Document")
business_req_file = st.file_uploader("Upload Business Requirements File", type=["txt", "pdf", "docx"])
business_req_text = ""
if business_req_file:
    business_req_text = business_req_file.read().decode("utf-8", errors="ignore")
else:
    business_req_text = st.text_area("Or paste the Business Requirements here:")


# #### Vendor History Upload (Step 2)

# In[256]:


st.subheader("Upload Vendor History File")
vendor_history_file = st.file_uploader("Upload Vendor History File", key="vendor", type=["txt", "csv"])
vendor_history_text = ""

if vendor_history_file:
    vendor_history_text = vendor_history_file.read().decode("utf-8", errors="ignore")

    # Read the data using pandas
    try:
        df = pd.read_csv(StringIO(vendor_history_text))

        # Store DataFrame in session state
        st.session_state.vendor_history_df = df

    except Exception as e:
        st.error(f"Error reading Vendor History file: {e}. Please ensure it is a valid TXT format.")
        st.session_state.vendor_history_df = None
else:
    vendor_history_text = st.text_area("Or paste the Vendor History here:", key="vendor_text")


# #### Bids File Upload (Step 5)

# In[258]:


st.subheader("Upload Bids File")
bids_file = st.file_uploader("Upload Bids File", key="bids", type=["txt", "pdf", "docx", "csv"])
bids_text = ""
if bids_file:
    bids_text = bids_file.read().decode("utf-8", errors="ignore")
else:
    bids_text = st.text_area("Or paste the Bids data here:", key="bids_text")


# ## 6. Processing and Output Generation

# In[260]:


st.header("Output Section")


# #### Step 1: Technical Requirements Document

# In[262]:


with st.expander("Step 1: Technical Requirements Document"):
    if st.button("Generate Technical Requirements"):
        if business_req_text.strip():
            with st.spinner("Generating Technical Requirements..."):
                tech_req_doc = tech_req_chain.run(business_req=business_req_text)
                st.session_state.tech_req_doc = tech_req_doc
                st.text_area("Technical Requirements Document:", value=tech_req_doc, height=300)
                st.download_button("Download Technical Requirements", tech_req_doc, file_name="Technical_Requirements.txt")
        else:
            st.error("Please provide the Business Requirements.")


# #### Step 2: Vendor Shortlisting

# In[264]:


with st.expander("Step 2: Vendor Shortlisting"):
    if st.button("Shortlist Vendors"):
        if st.session_state.tech_req_doc and st.session_state.vendor_history_df is not None:
            with st.spinner("Shortlisting Vendors..."):
                # 0.  Print column names for debugging
                #st.write("Vendor History DataFrame Columns:", list(st.session_state.vendor_history_df.columns))

                # 1. Get list of all unique vendors from LLM
                vendor_names = vendor_shortlist_chain.run(
                    tech_req=st.session_state.tech_req_doc,
                    vendor_history=vendor_history_text
                )
                # Splitting the comma-separated string into a list
                vendor_names_list = [name.strip() for name in vendor_names.split(',')]

                # 2. Calculate Composite Scores
                vendor_scores = {}
                df = st.session_state.vendor_history_df
                for vendor in vendor_names_list:
                    try:
                        vendor_data = df[df['Vendor_name'] == vendor]

                        if not vendor_data.empty:  # Check if the vendor data exists
                            avg_delivery = vendor_data['Delivery_punctuality'].mean()
                            avg_quality = vendor_data['Quality_of_goods'].mean()
                            avg_contract = vendor_data['Contract_term_compliance'].mean()
                            composite_score = (avg_delivery + avg_quality + avg_contract) / 3
                            vendor_scores[vendor] = {
                                "composite": composite_score,
                                "contract": avg_contract,
                                "quality": avg_quality,
                                "delivery": avg_delivery
                            }
                    except KeyError as e:
                        st.error(f"KeyError: {e}. Please ensure the Vendor History file has a column named 'Vendor_name', 'Delivery_punctuality', 'Quality_of_goods', and 'Contract_term_compliance'")
                        break  # Stop processing to avoid further errors

                # 3. Sort Vendors Based on Composite Score and Tie-Breaking
                sorted_vendors = sorted(vendor_scores.items(),
                                        key=lambda item: (item[1]['composite'],
                                                           item[1]['contract'],
                                                           item[1]['quality'],
                                                           item[1]['delivery']),
                                        reverse=True)

                # 4. Select Top Two Vendors
                top_two_vendors = [vendor[0] for vendor in sorted_vendors[:2]]
                st.session_state.shortlisted_vendors = ", ".join(top_two_vendors)

                st.text_area("Shortlisted Vendors:", value=st.session_state.shortlisted_vendors, height=200)
                st.download_button("Download Vendor Shortlist", st.session_state.shortlisted_vendors, file_name="Vendor_Shortlist.txt")

        else:
            st.error("Ensure Technical Requirements and Vendor History are provided and in the correct format.")


# #### Step 3: Tender Document & RFP 

# In[266]:


with st.expander("Step 3: Tender Document & RFP"):
    if st.button("Generate Tender Document"):
        if st.session_state.tech_req_doc and business_req_text.strip():
            with st.spinner("Generating Tender Document..."):
                tender_doc = tender_doc_chain.run(
                    tech_req=st.session_state.tech_req_doc,
                    business_req=business_req_text
                )
                st.session_state.tender_doc = tender_doc
                st.text_area("Tender Document & RFP:", value=tender_doc, height=300)
                st.download_button("Download Tender Document", tender_doc, file_name="Tender_Document.txt")
        else:
            st.error("Ensure Business Requirements and Technical Requirements are provided.")


# #### Step 4: Tender Email Generation

# In[268]:


with st.expander("Step 4: Tender Email Generation"):
    if st.button("Generate Tender Email"):
        if st.session_state.shortlisted_vendors and st.session_state.tender_doc:
            with st.spinner("Generating Tender Email..."):
                tender_email = tender_email_chain.run(
                    shortlisted_vendors=st.session_state.shortlisted_vendors,
                    tender_doc=st.session_state.tender_doc
                )
                st.session_state.tender_email = tender_email
                st.text_area("Tender Email:", value=tender_email, height=200)
                st.download_button("Download Tender Email", tender_email, file_name="Tender_Email.txt")
        else:
            st.error("Ensure Vendor Shortlist and Tender Document are generated.")


# #### Step 5: Bid Evaluation

# In[270]:


with st.expander("Step 5: Bid Evaluation"):
    if st.button("Evaluate Bids"):
        if bids_text.strip() and st.session_state.shortlisted_vendors:
            with st.spinner("Filtering Bids for Top Vendors..."):
                # Extract bids from shortlisted vendors using LLM
                extract_bids_prompt = PromptTemplate(
                    input_variables=["shortlisted_vendors", "bids_data"],
                    template="""
You are a PC hardware and procurement expert. We have the following vendor bids (in JSON-like form):
{bids_data}

Please qualitatively evaluate these bids based on:
- CPU generation (e.g., i7 12th gen vs i7 11th gen, Ryzen 7 4000 vs Ryzen 7 5000)
- Price
- RAM size
- Storage type & capacity (SSD vs HDD)
- OS (Windows 10 vs Windows 11)
- Overall value for money

Rank them from best to worst, providing a concise rationale. Identify the top 2 bids you recommend. 
Respond with bullet points and a final summary. Temperature 0.5

"""
                )
                extract_bids_chain = LLMChain(llm=llm, prompt=extract_bids_prompt)

                shortlisted_vendors_str = st.session_state.shortlisted_vendors  # Access shortlisted vendors from session state

                filtered_bids = extract_bids_chain.run(
                    shortlisted_vendors=shortlisted_vendors_str,
                    bids_data=bids_text
                )

                st.session_state.top_two_bids = filtered_bids

                with st.spinner("Evaluating Bids..."):
                    bid_evaluation = bid_evaluation_chain.run(bids_data=st.session_state.top_two_bids)
                    st.session_state.bid_evaluation = bid_evaluation
                    st.text_area("Bid Evaluation Report:", value=bid_evaluation, height=300)
                    st.download_button("Download Bid Evaluation", bid_evaluation, file_name="Bid_Evaluation.txt")
        else:
            st.error("Please provide the Bids data and ensure Vendor Shortlisting is complete.")


# #### Step 6: Negotiation Strategy & BATNA

# In[272]:


with st.expander("Step 6: Negotiation Strategy & BATNA"):
    if st.button("Generate Negotiation Strategy"):
        if st.session_state.bid_evaluation:
            with st.spinner("Generating Negotiation Strategy..."):
                negotiation_strategy = negotiation_strategy_chain.run(
                    top_two_bids=st.session_state.bid_evaluation
                )
                st.session_state.negotiation_strategy = negotiation_strategy
                st.text_area("Negotiation Strategy & BATNA:", value=negotiation_strategy, height=300)
                st.download_button("Download Negotiation Strategy", negotiation_strategy, file_name="Negotiation_Strategy.txt")
        else:
            st.error("Ensure Bid Evaluation is completed.")


# #### Step 7: Risk Assessment Report

# In[274]:


with st.expander("Step 7: Risk Assessment Report"):
    if st.button("Generate Risk Assessment"):
        if st.session_state.negotiation_strategy and st.session_state.top_two_bids.strip():
            with st.spinner("Generating Risk Assessment Report..."):
                risk_assessment = risk_assessment_chain.run(
                    negotiation_strategy=st.session_state.negotiation_strategy,
                    bid_data=st.session_state.top_two_bids
                )
                st.session_state.risk_assessment = risk_assessment
                st.text_area("Risk Assessment Report:", value=risk_assessment, height=300)
                st.download_button("Download Risk Assessment", risk_assessment, file_name="Risk_Assessment.txt")
        else:
            st.error("Ensure Negotiation Strategy and Bids data are provided.")


# #### Step 8: Contract Document Generation

# In[276]:


with st.expander("Step 8: Contract Document Generation"):
    if st.button("Generate Contract Document"):
        if st.session_state.risk_assessment:
            with st.spinner("Generating Contract Document..."):
                contract_doc = contract_doc_chain.run(
                    risk_assessment=st.session_state.risk_assessment
                )
                st.session_state.contract_doc = contract_doc
                st.text_area("Contract Document:", value=contract_doc, height=300)
                st.download_button("Download Contract Document", contract_doc, file_name="Contract_Document.txt")
        else:
            st.error("Ensure Risk Assessment is completed.")


# ## 7. Adding a Fixed Footer

# In[278]:


st.markdown("""
    <style>
        .fixed-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #555;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="fixed-footer">
        TransGlobal Industries Procurement Automation System | Powered by LangChain & Google Gemini Al
    </div>
    """, unsafe_allow_html=True)


# In[ ]:





# In[ ]:





