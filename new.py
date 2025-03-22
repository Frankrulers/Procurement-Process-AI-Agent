#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import pandas as pd
from io import BytesIO
import google.generativeai as genai

##############################################################################
# 1. Configure Google Generative AI (Gemini) 
##############################################################################
api_key = os.getenv("GOOGLE_API_KEY")  # Ensure you have this set in your environment
if not api_key:
    st.error("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=api_key)  # Use the API key from the environment

def get_model():
    """
    Return the Gemini 2.0 flash model instance.
    """
    return genai.GenerativeModel("gemini-2.0-flash")

##############################################################################
# 2. Streamlit Page Configuration & Styling
##############################################################################
st.set_page_config(page_title="Procurement Automation with Gemini", layout="wide")
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(to right, #4880EC, #019CAD);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: #f0f0f0;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            font-size: 0.8em;
        }
    </style>
    <div class="main-header">
        <h1>Procurement & Tender Evaluation Automation</h1>
        <p>Using Google Gemini 2.0-flash for LLM-based Document Generation</p>
    </div>
""", unsafe_allow_html=True)

##############################################################################
# 3. Step Functions
##############################################################################

# --------------------------------------------------
# Step 1: Identify Business Requirements
# --------------------------------------------------
def step1_business_requirements():
    st.header("Step 1: Identify Business Requirements")
    business_req = st.text_area(
        "Enter the Business Requirements (e.g., needs for new servers, software, or technical assets):",
        value=st.session_state.get("business_req", "")
    )
    
    if st.button("Save & Next (Step 1)"):
        if not business_req.strip():
            st.error("Please enter the business requirements.")
        else:
            st.session_state["business_req"] = business_req

            # Directly store a doc (no LLM used here, we only store user input).
            doc_text = f"Business Requirements Document:\n\n{business_req}"
            st.session_state["business_doc"] = doc_text

            st.success("Business requirements saved.")
            with st.expander("Preview of Generated Business Requirements Document"):
                st.text(doc_text)

            st.download_button(
                label="Download Business Requirements Document",
                data=doc_text,
                file_name="business_requirements.txt",
                mime="text/plain"
            )

# --------------------------------------------------
# Step 2: Convert Business Requirements into Technical Requirements
# --------------------------------------------------
def step2_technical_requirements():
    st.header("Step 2: Convert Business Requirements into Technical Requirements")
    if "business_req" not in st.session_state:
        st.error("Please complete Step 1 first.")
        return

    business_req = st.session_state["business_req"]
    
    if st.button("Generate Technical Requirements via LLM (Step 2)"):
        try:
            model = get_model()
            prompt = f"""
    Transform the following business requirements into a detailed technical requirements document:
    
    {business_req}
    
    The technical requirements document should include:
    
    1. A header with project title and date
    2. Numbered sections for different requirement categories
    3. For each requirement, include:
       * The specific technical requirement
       * The business purpose it serves
    
    Format the document in Markdown with clear headings, bullet points, and proper structure.
    Each requirement should be specific, measurable, achievable, relevant, and time-bound (SMART).
    
    Ensure all functional and non-functional requirements are covered, including:
    - System architecture
    - Performance specifications
    - Integration requirements
    - Security requirements
    - User interface specifications
    - Data management requirements
    - Any relevant standards or compliance needs
    """
    
            response = model.generate_content(prompt)
            st.session_state["technical_req"] = response.text
            st.success("Technical requirements generated via Gemini.")
        except Exception as e:
            st.error(f"Error calling Gemini for technical requirements: {e}")
            
    # If we already have technical requirements in session_state, display & download them
    if "technical_req" in st.session_state:
        with st.expander("Preview of Generated Technical Requirements Document"):
            st.text(st.session_state["technical_req"])

        st.download_button(
            label="Download Technical Requirements Document",
            data=st.session_state["technical_req"],
            file_name="technical_requirements.txt",
            mime="text/plain"
        )


# --------------------------------------------------
# Step 3: Develop Request for Proposal (RFP)
# --------------------------------------------------
def step3_rfp():
    st.header("Step 3: Develop Request for Proposal (RFP)")
    if "technical_req" not in st.session_state:
        st.error("Please complete Step 2 first.")
        return

    technical_req = st.session_state["technical_req"]

    if st.button("Generate RFP Document via LLM (Step 3)"):
        try:
            model = get_model()
            prompt = f"""
Using the following technical requirements, generate an RFP (Request for Proposal) document that clearly
states the technical and performance criteria potential suppliers must meet. Provide a professional and
comprehensive structure.

Technical Requirements:
{technical_req}

Output a well-structured RFP with clear sections and bullet points.
            """
            response = model.generate_content(prompt)
            rfp_doc = response.text
            st.session_state["rfp_doc"] = rfp_doc
            st.success("RFP document generated via Gemini.")
        except Exception as e:
            st.error(f"Error generating RFP: {e}")

    if "rfp_doc" in st.session_state:
        with st.expander("Preview of Generated RFP Document"):
            st.text(st.session_state["rfp_doc"])

        st.download_button(
            label="Download RFP Document",
            data=st.session_state["rfp_doc"],
            file_name="rfp_document.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Step 4: Identify Suitable Vendors
# --------------------------------------------------
def step4_vendor_selection():
    st.header("Step 4: Identify Suitable Vendors")
    try:
        # File uploader to load vendor_history.csv
        if "vendor_history" not in st.session_state:
            uploaded_vendor = st.file_uploader("Upload vendor_history.csv", type=["csv"])
            if uploaded_vendor is not None:
                vendor_df = pd.read_csv(uploaded_vendor)
                st.session_state["vendor_history"] = vendor_df
            else:
                st.info("Please upload the vendor_history.csv file to proceed.")
                return
        else:
            vendor_df = st.session_state["vendor_history"]

        st.write("Vendor History Data:")
        st.dataframe(vendor_df)

        # Evaluate vendors by computing a simple average of numeric columns
        numeric_cols = vendor_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in vendor data for evaluation.")
            return

        vendor_df["score"] = vendor_df[numeric_cols].mean(axis=1)
        top_vendors = vendor_df.nlargest(2, "score")

        st.write("Top 2 Vendors Based on Weighted Evaluation:")
        st.dataframe(top_vendors)

        if st.button("Save & Next (Step 4)"):
            st.session_state["selected_vendors"] = top_vendors
            selected_vendors_csv = top_vendors.to_csv(index=False)
            st.success("Selected vendors saved.")

            with st.expander("Preview of Selected Vendors Document"):
                st.text(selected_vendors_csv)

            st.download_button(
                label="Download Selected Vendors Document",
                data=selected_vendors_csv,
                file_name="selected_vendors.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing vendor data: {e}")

# --------------------------------------------------
# Step 5: Issue RFP and Tender Documents
# --------------------------------------------------
def step5_issue_tender():
    st.header("Step 5: Issue RFP and Tender Documents")
    if "rfp_doc" not in st.session_state or "selected_vendors" not in st.session_state:
        st.error("Please complete Steps 3 and 4 first.")
        return

    rfp_doc = st.session_state["rfp_doc"]
    top_vendors = st.session_state["selected_vendors"]

    if st.button("Generate Tender Document via LLM (Step 5)"):
        try:
            model = get_model()
            prompt = f"""
We have the following RFP content:

{rfp_doc}

We have shortlisted these vendors:
{top_vendors.to_csv(index=False)}

Generate a professional tender document that encapsulates the RFP requirements and addresses
these shortlisted vendors. The tone should be formal and ready for direct communication.
            """
            response = model.generate_content(prompt)
            tender_doc = response.text
            st.session_state["tender_doc"] = tender_doc
            st.success("Tender document generated via Gemini.")
        except Exception as e:
            st.error(f"Error generating tender document: {e}")

    if "tender_doc" in st.session_state:
        with st.expander("Preview of Generated Tender Document"):
            st.text(st.session_state["tender_doc"])

        st.download_button(
            label="Download Tender Document",
            data=st.session_state["tender_doc"],
            file_name="tender_document.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Step 6: Receive and Evaluate Bids (Qualitative LLM-based, temperature=0.5)
# --------------------------------------------------
def step6_evaluate_bids():
    st.header("Step 6: Receive and Evaluate Bids (Qualitative)")
    if "selected_vendors" not in st.session_state:
        st.error("Please complete Step 4 first.")
        return

    # File uploader to load bid.csv
    if "bids" not in st.session_state:
        uploaded_bid = st.file_uploader("Upload bid.csv", type=["csv"])
        if uploaded_bid is not None:
            bids_df = pd.read_csv(uploaded_bid)
            st.session_state["bids"] = bids_df
        else:
            st.info("Please upload the bid.csv file to proceed.")
            return
    else:
        bids_df = st.session_state["bids"]

    st.write("All Bids Data:")
    st.dataframe(bids_df)

    # Filter bids based on the selected vendors
    selected_vendors = st.session_state["selected_vendors"]
    if "vendor_id" in selected_vendors.columns and "vendor_id" in bids_df.columns:
        filtered_bids = bids_df[bids_df["vendor_id"].isin(selected_vendors["vendor_id"])]
    elif "vendor_name" in selected_vendors.columns and "vendor_name" in bids_df.columns:
        filtered_bids = bids_df[bids_df["vendor_name"].isin(selected_vendors["vendor_name"])]
    else:
        st.warning("Vendor identifiers do not match; evaluating all bids.")
        filtered_bids = bids_df

    st.write("Filtered Bids for Selected Vendors:")
    st.dataframe(filtered_bids)

    # Convert each row into a dict to feed into the LLM
    bid_list = filtered_bids.to_dict(orient="records")

    if st.button("Perform Qualitative Evaluation via LLM (Step 6)", key="qual_eval"):
        try:
            model = get_model()
            # Build a prompt that describes the hardware specs, price, OS, CPU gen, etc.
            prompt = f"""
You are a PC hardware and procurement expert. We have the following vendor bids (in JSON-like form):
{bid_list}

Please qualitatively evaluate these bids based on:
- CPU generation (e.g., i7 12th gen vs i7 11th gen)
- Price
- RAM size
- Storage type & capacity (SSD vs HDD)
- OS (Windows 10 vs Windows 11)
- Overall value for money

Rank them from best to worst, providing a concise rationale. Identify the top 2 bids you recommend. 
Respond with bullet points and a final summary. Temperature 0.5
            """
            response = model.generate_content(
                prompt)
            evaluation_text = response.text
            st.session_state["qualitative_evaluation"] = evaluation_text
            st.success("Qualitative evaluation complete via Gemini.")
        except Exception as e:
            st.error(f"Error in qualitative LLM evaluation: {e}")

    if "qualitative_evaluation" in st.session_state:
        with st.expander("Preview of Qualitative Evaluation"):
            st.markdown(st.session_state["qualitative_evaluation"])

        st.download_button(
            label="Download Qualitative Evaluation",
            data=st.session_state["qualitative_evaluation"],
            file_name="qualitative_evaluation.txt",
            mime="text/plain"
        )

        if st.button("Save & Next (Step 6)"):
            st.session_state["evaluated_bids_doc"] = st.session_state["qualitative_evaluation"]
            st.success("Bids evaluation (qualitative) saved.")

# --------------------------------------------------
# Step 7: Select Top Bids
# --------------------------------------------------
def step7_select_top_bids():
    st.header("Step 7: Select Top Bids")
    if "qualitative_evaluation" not in st.session_state:
        st.error("Please complete Step 6 (qualitative evaluation).")
        return

    st.write("LLM's Qualitative Evaluation of Bids:")
    st.markdown(st.session_state["qualitative_evaluation"])

    st.info("In a real system, you'd parse the LLM's output to identify top 2 bids automatically.")
    # For demonstration, we simply finalize "top 2" from the text itself.

    if st.button("Confirm Top 2 Bids"):
        # This might be extracted from the LLM text in a real app
        sample_top_bids = (
            "Top 2 Bids (Based on LLM’s Recommendation):\n\n"
            "(Example) 1) Vendor A: i7 12th gen, 16GB RAM, Windows 11\n"
            "(Example) 2) Vendor B: i7 11th gen, 16GB RAM, Windows 10\n"
        )
        st.session_state["top_bids"] = sample_top_bids
        st.success("Top 2 Bids confirmed from LLM output.")

    if "top_bids" in st.session_state:
        with st.expander("Preview of Top 2 Bids Document"):
            st.text(st.session_state["top_bids"])

        st.download_button(
            label="Download Top Bids Document",
            data=st.session_state["top_bids"],
            file_name="top_bids.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Step 8: Negotiation Strategy & BATNA Analysis
# --------------------------------------------------
def step8_negotiation_strategy():
    st.header("Step 8: Negotiation Strategy & BATNA Analysis")
    if "top_bids" not in st.session_state:
        st.error("Please complete Step 7 first.")
        return

    if st.button("Generate Negotiation Strategy via LLM"):
        try:
            model = get_model()
            prompt = f"""
We have selected the following top 2 bids:
{st.session_state['top_bids']}

Develop a short Negotiation Strategy & BATNA analysis. Provide:
1. Main negotiation levers (price, performance, additional services)
2. A recommended BATNA in case negotiations fail
3. Recommended approach for each vendor
            """
            response = model.generate_content(prompt)
            st.session_state["negotiation_strategy"] = response.text
            st.success("Negotiation strategy generated via Gemini.")
        except Exception as e:
            st.error(f"Error generating negotiation strategy: {e}")

    if "negotiation_strategy" in st.session_state:
        with st.expander("Preview of Negotiation Strategy Document"):
            st.text(st.session_state["negotiation_strategy"])

        st.download_button(
            label="Download Negotiation Strategy Document",
            data=st.session_state["negotiation_strategy"],
            file_name="negotiation_strategy.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Step 9: Negotiate with Preferred Vendor
# --------------------------------------------------
def step9_negotiate_vendor():
    st.header("Step 9: Negotiate with Preferred Vendor")
    if "negotiation_strategy" not in st.session_state:
        st.error("Please complete Step 8 first.")
        return

    if st.button("Generate Negotiation Summary via LLM"):
        try:
            model = get_model()
            prompt = f"""
We have this negotiation strategy:
{st.session_state['negotiation_strategy']}

Summarize how we should approach direct negotiations with the preferred vendor
to secure the best terms. Provide a concise set of action items.
            """
            response = model.generate_content(prompt)
            st.session_state["negotiation_summary"] = response.text
            st.success("Negotiation summary created via Gemini.")
        except Exception as e:
            st.error(f"Error generating negotiation summary: {e}")

    if "negotiation_summary" in st.session_state:
        with st.expander("Preview of Negotiation Summary Document"):
            st.text(st.session_state["negotiation_summary"])

        st.download_button(
            label="Download Negotiation Summary Document",
            data=st.session_state["negotiation_summary"],
            file_name="negotiation_summary.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Step 10: Risk Assessment & Final Contract Creation
# --------------------------------------------------
def step10_final_contract():
    st.header("Step 10: Risk Assessment & Final Contract Creation")
    if "negotiation_summary" not in st.session_state:
        st.error("Please complete Step 9 first.")
        return

    if st.button("Generate Risk Assessment & Final Contract via LLM"):
        try:
            model = get_model()
            prompt = f"""
We have concluded negotiations with the preferred vendor. Summarize potential risks
(supplier reliability, quality issues, delivery delays) and their mitigation strategies.
Then, draft the final contract with key clauses: performance guarantees, penalty clauses,
dispute resolution, etc.

Negotiation Summary:
{st.session_state['negotiation_summary']}
            """
            response = model.generate_content(prompt)
            final_contract = response.text
            st.session_state["final_contract"] = final_contract
            st.success("Final contract generated via Gemini.")
        except Exception as e:
            st.error(f"Error generating final contract: {e}")

    if "final_contract" in st.session_state:
        with st.expander("Preview of Final Contract Document"):
            st.text(st.session_state["final_contract"])

        st.download_button(
            label="Download Final Contract Document",
            data=st.session_state["final_contract"],
            file_name="final_contract.txt",
            mime="text/plain"
        )

##############################################################################
# 4. Sidebar Navigation for the 10-Step Process
##############################################################################
steps = [
    "Step 1: Identify Business Requirements",
    "Step 2: Convert to Technical Requirements",
    "Step 3: Develop RFP",
    "Step 4: Identify Suitable Vendors",
    "Step 5: Issue RFP and Tender Documents",
    "Step 6: Receive and Evaluate Bids (Qualitative)",
    "Step 7: Select Top Bids",
    "Step 8: Negotiation Strategy & BATNA Analysis",
    "Step 9: Negotiate with Preferred Vendor",
    "Step 10: Risk Assessment & Final Contract Creation"
]

choice = st.sidebar.radio("Select Process Step", steps)

if choice == steps[0]:
    step1_business_requirements()
elif choice == steps[1]:
    step2_technical_requirements()
elif choice == steps[2]:
    step3_rfp()
elif choice == steps[3]:
    step4_vendor_selection()
elif choice == steps[4]:
    step5_issue_tender()
elif choice == steps[5]:
    step6_evaluate_bids()
elif choice == steps[6]:
    step7_select_top_bids()
elif choice == steps[7]:
    step8_negotiation_strategy()
elif choice == steps[8]:
    step9_negotiate_vendor()
elif choice == steps[9]:
    step10_final_contract()

##############################################################################
# 5. Footer
##############################################################################
st.markdown("""
    <div class="footer">
        Procurement Automation App © 2025 | Powered by Google Gemini 2.0-flash
    </div>
""", unsafe_allow_html=True)

