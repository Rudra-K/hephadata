from dataclasses import dataclass, field
import random
import uuid
import pandas as pd

from .knowledge_base import DOMAIN_KNOWLEDGE_CONFIG


@dataclass
class Profile:
    """Represents the starting point for a simulated person's career."""
    profile_id: str = field(default_factory=lambda: f"prof_{uuid.uuid4().hex[:8]}")
    persona: str = 'new_grad'
    primary_domain: str = 'Technology'
    aptitudes: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    initial_skills: list[str] = field(default_factory=list)
    location_city: str = 'Pune'

@dataclass
class CareerEvent:
    """Represents a single event or job in a career path."""
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    profile_id: str = ""
    career_year: float = 0.0
    title: str = ""
    work_type: str = "In-Office"
    required_skills: list[str] = field(default_factory=list)
    skills_acquired: list[str] = field(default_factory=list)
    salary_lpa: float = 0.0
    stability_rating: int = 3

class CareerPathGenerator:
    """
    Generates a realistic dataset of career trajectories.
    """
    def __init__(self):
        self._knowledge_config = DOMAIN_KNOWLEDGE_CONFIG

    def _create_profiles(self, n_profiles: int, persona_mix: dict) -> list[Profile]:
        """Creates the initial population of profiles."""
        population = []
        for persona, domains in persona_mix.items():
            for domain, percentage in domains.items():
                num_to_generate = int(n_profiles * percentage)
                domain_info = self._knowledge_config.get(domain, {})
                
                for _ in range(num_to_generate):
                    # --- FIX IS HERE ---
                    # Get the full list of available aptitudes and interests
                    aptitude_pool = domain_info.get('aptitudes', [])
                    interest_pool = domain_info.get('interests', [])

                    # Safely choose a random number (k) of items to sample
                    # k will be between 1 and the actual length of the pool
                    num_aptitudes = random.randint(1, len(aptitude_pool)) if aptitude_pool else 0
                    num_interests = random.randint(1, len(interest_pool)) if interest_pool else 0

                    # Perform the sampling with the safe k value
                    aptitudes = random.sample(aptitude_pool, k=num_aptitudes)
                    interests = random.sample(interest_pool, k=num_interests)
                    # --- END FIX ---
                    
                    population.append(Profile(
                        persona=persona, 
                        primary_domain=domain,
                        aptitudes=aptitudes,
                        interests=interests,
                        initial_skills=domain_info.get('new_grad_skills', [])
                    ))
        random.shuffle(population)
        return population

    def _simulate_career_for_profile(self, profile: Profile, sim_years: int) -> list[CareerEvent]:
        """Simulates a career path for a single profile."""
        career_path = []
        current_year = 0
        
        # Start with a random entry-level role from their domain
        current_role_name = random.choice(self._knowledge_config[profile.primary_domain]['entry_roles'])
        
        while current_year < sim_years:
            role_info = self._knowledge_config[profile.primary_domain][current_role_name]
            
            # Simulate how long they stay in this role (e.g., 2 to 4 years)
            duration = round(random.uniform(2.0, 4.0), 1)
            
            if current_year + duration > sim_years:
                duration = sim_years - current_year
            
            career_path.append(CareerEvent(
                profile_id=profile.profile_id,
                career_year=round(current_year, 1),
                title=current_role_name,
                work_type=random.choice(['In-Office', 'Hybrid', 'Remote']),
                required_skills=role_info['required_skills'],
                skills_acquired=role_info['acquired_skills'],
                salary_lpa=round(random.uniform(*role_info['salary_range_lpa']), 1),
                stability_rating=random.randint(*role_info['stability_rating_range'])
            ))
            
            current_year += duration
            
            # Decide on the next role (promotion/change)
            if role_info['possible_next_roles']:
                current_role_name = random.choice(role_info['possible_next_roles'])
            else:
                break # End of career path

        return career_path

    def generate(self, n_profiles: int, persona_mix: dict, sim_years: int = 15):
        """Generates both the profiles and their career event histories."""
        profiles = self._create_profiles(n_profiles, persona_mix)
        all_events = []

        print(f"\nSimulating career paths for {len(profiles)} profiles over {sim_years} years...")
        for profile in profiles:
            career_events = self._simulate_career_for_profile(profile, sim_years)
            all_events.extend(career_events)
        
        print("Simulation complete.")
        profiles_df = pd.DataFrame([vars(p) for p in profiles])
        events_df = pd.DataFrame([vars(e) for e in all_events])
        return profiles_df, events_df

# --- Testing Block ---
if __name__ == '__main__':
    generator = CareerPathGenerator()
    
    total_profiles = 10
    profile_mix_config = {'new_grad': {'Technology': 1.0}}

    profiles_data, events_data = generator.generate(
        n_profiles=total_profiles,
        persona_mix=profile_mix_config
    )

    print("\n--- Generated Profiles DataFrame ---")
    print(profiles_data.head())
    
    print("\n--- Sample Career Path for one Profile ---")
    sample_profile_id = profiles_data.iloc[0]['profile_id']
    print(events_data[events_data['profile_id'] == sample_profile_id].to_string())

    # --- NEW SUMMARY SECTION ---
    print("\n--------------------")
    print("--- FINAL SUMMARY ---")
    print(f"Total profiles generated and saved: {profiles_data.shape[0]}")
    print(f"Total career events generated and saved: {events_data.shape[0]}")
    print("--------------------")