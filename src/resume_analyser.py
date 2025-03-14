import spacy # type: ignore
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai # type: ignore

class ResumeAnalyzer:
    def __init__(self):
        # Load pre-trained models
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        
        # Set OpenAI API key (replace with your actual key)
        openai.api_key = 'your_openai_api_key_here'

    def extract_keywords(self, text):
        # Extract key skills and entities
        doc = self.nlp(text)
        keywords = [ent.text for ent in doc.ents] + [token.text for token in doc if token.pos_ in ['NOUN', 'VERB']]
        return list(set(keywords))

    def match_job_description(self, resume_text, job_description):
        # Calculate resume-job match percentage
        resume_keywords = set(self.extract_keywords(resume_text))
        job_keywords = set(self.extract_keywords(job_description))
        
        match_score = len(resume_keywords.intersection(job_keywords)) / len(job_keywords) * 100
        return round(match_score, 2)

    def suggest_improvements(self, resume_text, job_description):
        # Use GPT for suggesting resume improvements
        prompt = f"""
        Analyze this resume against the job description and suggest improvements:
        
        Resume: {resume_text}
        Job Description: {job_description}
        
        Provide specific suggestions to enhance the resume's match and impact.
        """
        
        try:
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                max_tokens=300
            )
            
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"

# Example usage
def main():
    analyzer = ResumeAnalyzer()
    
    # Sample resume and job description for testing
    sample_resume = "Software engineer with 3 years of experience in Python and machine learning"
    sample_job_description = "We are seeking a Python developer with machine learning expertise"
    
    # Extract keywords
    keywords = analyzer.extract_keywords(sample_resume)
    print("Extracted Keywords:", keywords)
    
    # Match job description
    match_score = analyzer.match_job_description(sample_resume, sample_job_description)
    print("Job Match Score:", match_score)
    
    # Get improvement suggestions
    improvements = analyzer.suggest_improvements(sample_resume, sample_job_description)
    print("Improvement Suggestions:", improvements)

# Only run main if this script is run directly
if __name__ == "__main__":
    main()