#FINALLLL

import google.generativeai as genai
import pandas as pd
import json
import re

# Configure API key
genai.configure(api_key='AIzaSyDo5wrwBLqsrxWs7crmX_MyPSCV-fXD_rA')

def generate_employee_batch(columns, departments, num_records):
    """
    Generate a batch of employee records using Gemini API.
    """
    prompt = f"""Generate a realistic JSON dataset for employee records with these specifications:

1. {num_records} unique employee records
2. Columns: {', '.join(columns)}
3. Departments: {', '.join(departments)}
4. Ensure:
    - Unique `Emp_No` for each record
    - Salary ranges appropriate to departments
    - Age distribution between 22-60
    - Names vary realistically
    - Balanced department distribution

### Output Rules:
- Output must be **ONLY a valid JSON array** of objects.
- Example:
[
    {{
        "Emp_No": 1,
        "Emp_Name": "Alice Smith",
        "Salary": 75000,
        "Age": 32,
        "Department": "IT"
    }},
    ...
]
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        match = re.search(r'\[\s*{.*}\s*\]', response.text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON array found in response.")
        
        json_data = match.group(0)
        return pd.DataFrame(json.loads(json_data))
    except json.JSONDecodeError as jde:
        print(f"JSON decoding error: {jde}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return pd.DataFrame()

def generate_employee_dataset(columns, departments, total_records, batch_size=100):
    """
    Generate the full employee dataset by requesting smaller batches.
    """
    all_batches = []
    for start in range(0, total_records, batch_size):
        remaining = min(batch_size, total_records - start)
        print(f"Generating batch: {start + 1} to {start + remaining}")
        batch_df = generate_employee_batch(columns, departments, remaining)
        if not batch_df.empty:
            batch_df['Emp_No'] += start  # Adjust employee numbers to ensure uniqueness
            all_batches.append(batch_df)
    
    if all_batches:
        return pd.concat(all_batches, ignore_index=True)
    return pd.DataFrame()

# Example usage
def main():
    columns = ['Emp_No', 'Emp_Name', 'Salary', 'Age', 'Department']
    departments = ['IT', 'Marketing', 'Sales', 'Finance', 'HR']
    total_records = 500

    employees_df = generate_employee_dataset(columns, departments, total_records)

    if not employees_df.empty:
        employees_df = employees_df.drop_duplicates(subset=['Emp_No'])
        employees_df.to_csv('emp.csv', index=False)
        
        print("Dataset Generation Successful:")
        print(f"Total Records: {len(employees_df)}")
        print("\nDataset Preview:")
        print(employees_df.head())
        
        print("\nDepartment Distribution:")
        print(employees_df['Department'].value_counts())
    else:
        print("Dataset generation failed.")

# Run the script
if __name__ == "__main__":
    main()
