DOMAIN_KNOWLEDGE_CONFIG = {
    'Technology': {
        # --- GENERAL DOMAIN INFO (Restored) ---
        'aptitudes': ['analytical', 'problem-solving', 'logical-thinking'],
        'interests': ['AI/ML', 'Cloud Computing', 'Gaming', 'Cybersecurity'],
        'new_grad_skills': ['Python', 'Git', 'SQL', 'Data Structures'],
        'entry_roles': ['Data Analyst', 'Junior Software Engineer'],
        # --- END GENERAL DOMAIN INFO ---

        # --- SPECIFIC ROLE INFO ---
        'Data Analyst': {
            'required_skills': ['SQL', 'Excel', 'Statistics'],
            'acquired_skills': ['Python (Pandas)', 'Tableau', 'PowerBI'],
            'salary_range_lpa': (4, 8),
            'stability_rating_range': (3, 5),
            'possible_next_roles': ['Senior Data Analyst', 'Data Scientist', 'Product Manager']
        },
        'Junior Software Engineer': {
            'required_skills': ['Python', 'Git', 'Data Structures'],
            'acquired_skills': ['AWS', 'Kubernetes', 'Docker', 'React'],
            'salary_range_lpa': (5, 9),
            'stability_rating_range': (3, 5),
            'possible_next_roles': ['Software Engineer', 'DevOps Engineer']
        },
        'Software Engineer': {
            'required_skills': ['AWS', 'Docker', 'System Design'],
            'acquired_skills': ['Microservices', 'CI/CD', 'Go'],
            'salary_range_lpa': (12, 25),
            'stability_rating_range': (4, 5),
            'possible_next_roles': ['Senior Software Engineer', 'Tech Lead']
        },
        'Senior Software Engineer': {
            'required_skills': ['Microservices', 'System Design', 'Performance Tuning'],
            'acquired_skills': ['Mentoring', 'Technical Debt Management', 'Advanced Caching'],
            'salary_range_lpa': (20, 45),
            'stability_rating_range': (5, 5),
            'possible_next_roles': ['Tech Lead']
        },
        'DevOps Engineer': {
            'required_skills': ['AWS', 'Docker', 'Linux', 'Bash Scripting'],
            'acquired_skills': ['Terraform', 'CI/CD Pipelines', 'Kubernetes'],
            'salary_range_lpa': (10, 22),
            'stability_rating_range': (4, 5),
            'possible_next_roles': ['Senior DevOps Engineer', 'Cloud Architect']
        },
        'Senior DevOps Engineer': {
            'required_skills': ['Kubernetes', 'Terraform', 'System Architecture'],
            'acquired_skills': ['Advanced Kubernetes', 'Security Best Practices'],
            'salary_range_lpa': (20, 40),
            'stability_rating_range': (4, 5),
            'possible_next_roles': [] 
        },
        'Cloud Architect': {
            'required_skills': ['AWS', 'Azure', 'GCP', 'System Design', 'Cost Management'],
            'acquired_skills': ['Multi-Cloud Strategy', 'Infrastructure as Code'],
            'salary_range_lpa': (25, 50),
            'stability_rating_range': (5, 5),
            'possible_next_roles': []
        },
        'Senior Data Analyst': {
            'required_skills': ['Python (Pandas)', 'Tableau', 'Business Acumen'],
            'acquired_skills': ['A/B Testing', 'Advanced Statistics', 'Storytelling'],
            'salary_range_lpa': (10, 18),
            'stability_rating_range': (4, 5),
            'possible_next_roles': []
        },
        'Data Scientist': {
            'required_skills': ['Python (Scikit-learn)', 'Machine Learning Theory', 'Statistics'],
            'acquired_skills': ['Deep Learning (TensorFlow/PyTorch)', 'MLOps', 'Causal Inference'],
            'salary_range_lpa': (15, 35),
            'stability_rating_range': (4, 5),
            'possible_next_roles': []
        },
        'Product Manager': {
            'required_skills': ['User Research', 'Roadmapping', 'Agile Methodologies'],
            'acquired_skills': ['Market Analysis', 'Go-to-Market Strategy', 'Stakeholder Management'],
            'salary_range_lpa': (18, 45),
            'stability_rating_range': (4, 5),
            'possible_next_roles': []
        },
        'Tech Lead': {
            'required_skills': ['System Architecture', 'Code Review', 'Mentoring'],
            'acquired_skills': ['Team Leadership', 'Project Scoping', 'Technical Strategy'],
            'salary_range_lpa': (30, 60),
            'stability_rating_range': (5, 5),
            'possible_next_roles': []
        }
    }
}