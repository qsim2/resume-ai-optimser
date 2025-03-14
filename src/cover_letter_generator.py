from transformers import pipeline

class CoverLetterGenerator:
    def __init__(self):
        self.generator = pipeline('text-generation', model='gpt2')

    def generate_cover_letter(self, job_description, resume_keywords):
        prompt = f"""
        Write a professional cover letter for a job with the following description:
        {job_description}

        Key skills to highlight: {', '.join(resume_keywords)}
        """
        
        cover_letter = self.generator(prompt, max_length=500)[0]['generated_text']
        return cover_letter
