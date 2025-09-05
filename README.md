# Agnos_recommender_system
The system contains 2 parts:
1. **Main**: The system recommend Top 5 of symptoms that the input contains gender, age, and symptoms
   The code:
   - **Main_process_data.py**: file .py for managing and processing the historical data, and save data to 'assets/Main_process_profile.csv'
   - **Main_recommend_system.py**: file.py for creating logic to recommend symptoms
   - **Main_symptom_recommender_screen.py**: file .py for representing interface (screen) that enter inputs(gender, age, and symptoms) and show output(Top 5 of symptoms)
2. **Addition**: The system recommend Top 5 of symptoms that the input contains gender, age, symptoms and details (duration, previous treatment)
   The code:
   - **Addition_process_data.py**: file .py for managing and processing the historical data, and save data to 'assets/Addition_process_profile.csv'
   - **Addition_recommend_system.py**: file .py for creating logic to recommend symptoms, entering inputs(gender, age, symptoms, duration, and previous treatment) and showing output(Top 5 of symptoms)

*** For **utils.py** file >> file contains the function that use both main and addition parts ***

### Run code for recommender system
- For main part: run 'python Main_symptom_recommender_screen.py'
- For addition part: run 'python Addition_recommend_system.py'

### Here is Document
> https://docs.google.com/presentation/d/1GWaEBczhTfPEXU4H_rzeSkJRGpcZplAej9UqB27ltYQ/edit?usp=sharing
