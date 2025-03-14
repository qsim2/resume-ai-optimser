import streamlit as st # type: ignore
import spacy # type: ignore
import torch # type: ignore
import PyPDF2 # type: ignore
import re

# Use a simpler text generation approach
def generate_suggestions(resume_text, job_description):
    """
    Generate resume suggestions using a rule-based and AI-inspired approach
    """
    # Extract key skills and experiences
    def extract_key_points(text):
        # Use simple regex and spaCy to extract key information
        doc = spacy.load('en_core_web_sm')(text)
        
        # Extract entities and nouns
        entities = [ent.text for ent in doc.ents]
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        
        return list(set(entities + nouns))

    # Compare resume and job description
    def analyze_match(resume_skills, job_description):
        job_doc = spacy.load('en_core_web_sm')(job_description)
        job_keywords = [token.text for token in job_doc if token.pos_ in ['NOUN', 'VERB']]
        
        match_count = len(set(resume_skills) & set(job_keywords))
        match_percentage = (match_count / len(job_keywords)) * 100 if job_keywords else 0
        
        return round(match_percentage, 2)

    # Generate improvement suggestions
    def create_suggestions(resume_skills, job_description, match_score):
        suggestions = []
        
        # Generic suggestions based on match score
        if match_score < 30:
            suggestions.append("🔴 Low match: Significant resume revision needed")
        elif match_score < 60:
            suggestions.append("🟠 Moderate match: Consider major updates")
        else:
            suggestions.append("🟢 Strong match: Minor refinements suggested")
        
        # Skill-based suggestions
        suggestions.extend([
            "1. Align resume keywords with job description",
            "2. Use industry-specific terminology",
            "3. Quantify achievements with metrics",
            "4. Highlight relevant skills prominently",
            "5. Tailor professional summary to job requirements"
        ])
        
        return "\n".join(suggestions)

    try:
        # Extract key skills from resume
        resume_skills = extract_key_points(resume_text)
        
        # Calculate match percentage
        match_score = analyze_match(resume_skills, job_description)
        
        # Generate suggestions
        suggestions = create_suggestions(resume_skills, job_description, match_score)
        
        return suggestions
    except Exception as e:
        return f"Suggestion generation error: {str(e)}"

class ResumeAnalyzer:
    def __init__(self):
        # Load SpaCy for keyword extraction
        self.nlp = spacy.load('en_core_web_sm')

    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file (UploadedFile): Streamlit uploaded PDF file
        
        Returns:
            str: Extracted text from PDF
        """
        try:
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean up extracted text
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {e}")
            return ""

    def match_job_description(self, resume_text, job_description):
        """Calculate resume-job description match percentage"""
        try:
            # Extract keywords
            resume_keywords = set(self.extract_keywords(resume_text))
            job_keywords = set(self.extract_keywords(job_description))
            
            # Calculate match
            if not job_keywords:
                return 0
            
            match_count = len(resume_keywords.intersection(job_keywords))
            match_percentage = (match_count / len(job_keywords)) * 100
            
            return round(match_percentage, 2)
        except Exception as e:
            st.error(f"Job description matching error: {e}")
            return 0

    def extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        try:
            doc = self.nlp(text)
            keywords = [
                token.text 
                for token in doc 
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                and not token.is_stop
                and len(token.text) > 2  # Avoid very short tokens
            ]
            return list(set(keywords))
        except Exception as e:
            st.warning(f"Keyword extraction error: {e}")
            return []

def main():
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="🚀 AI Resume Optimizer", 
        page_icon="📄",
        layout="wide"
    )

    # Title and Introduction
    st.title("🚀 AI Resume Optimizer")
    st.markdown("""
    ### Enhance Your Resume with AI-Powered Insights
    - Upload your resume (PDF only)
    - Add the job description
    - Receive personalized optimization suggestions
    """)

    # Initialize analyzer
    analyzer = ResumeAnalyzer()

    # Sidebar Input Section
    with st.sidebar:
        st.header("Resume Analysis")
        
        # PDF Resume Upload
        uploaded_resume = st.file_uploader(
            "Upload Resume (PDF only)", 
            type=['pdf'],
            help="Upload your resume as a PDF"
        )
        
        # Job Description Input
        job_description = st.text_area(
            "Paste Job Description", 
            height=250, 
            help="Copy and paste the full job description"
        )
        
        # Analyze Button
        analyze_button = st.button("Analyze Resume", type="primary")

    # Analysis Section
    if analyze_button:
        # Input Validation
        if not uploaded_resume:
            st.warning("Please upload a resume PDF")
            return
        
        if not job_description:
            st.warning("Please provide a job description")
            return

        try:
            # Extract text from uploaded PDF
            resume_text = analyzer.extract_text_from_pdf(uploaded_resume)
            
            if not resume_text:
                st.error("Could not extract text from the PDF. Please ensure it's a readable PDF.")
                return
            
            # Results Columns
            col1, col2 = st.columns(2)
            
            # Match Score Visualization
            with col1:
                st.subheader("Job Match Score")
                match_score = analyzer.match_job_description(resume_text, job_description)
                
                # Color-coded Match Score
                if match_score < 30:
                    st.error(f"Match Score: {match_score}% 🔴 (Low)")
                elif match_score < 60:
                    st.warning(f"Match Score: {match_score}% 🟠 (Average)")
                else:
                    st.success(f"Match Score: {match_score}% 🟢 (Strong)")
            
            # Improvement Suggestions
            with col2:
                st.subheader("Improvement Suggestions")
                suggestions = generate_suggestions(resume_text, job_description)
                st.info(suggestions)
        
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    main()