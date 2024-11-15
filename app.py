from flask import Flask, render_template, request, jsonify, send_file
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col, avg, sum, count, min, max
import os
import pandas as pd
import io

app = Flask(__name__)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Employee Dashboard") \
    .getOrCreate()

# Define schema
schema = StructType([
    StructField("Emp_No", IntegerType(), False),
    StructField("Emp_Name", StringType(), False),
    StructField("Salary", IntegerType(), False),
    StructField("Age", IntegerType(), False),
    StructField("Department", StringType(), False),
])

# Load data function with error handling
def load_employee_data():
    try:
        # Get absolute path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'employees.csv')
        
        print(f"Loading CSV from: {csv_path}")  # Debug print
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at: {csv_path}")  # Debug print
            return None
            
        df = spark.read \
            .format("csv") \
            .schema(schema) \
            .option("header", "true") \
            .load(csv_path)
            
        # Debug print
        print(f"Loaded {df.count()} records from CSV")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")  # Debug print
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/employees')
def get_employees():
    try:
        df = load_employee_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
            
        # Convert to list of dictionaries for JSON serialization
        employees = df.collect()
        employee_list = [
            {
                "Emp_No": row["Emp_No"],
                "Emp_Name": row["Emp_Name"],
                "Salary": row["Salary"],
                "Age": row["Age"],
                "Department": row["Department"]
            }
            for row in employees
        ]
        return jsonify(employee_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/department_stats')
def get_department_stats():
    try:
        df = load_employee_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
            
        stats = df.groupBy("Department").agg(
            count("*").alias("count"),
            avg("Salary").alias("avg_salary"),
            sum("Salary").alias("total_salary"),
            min("Salary").alias("min_salary"),
            max("Salary").alias("max_salary"),
            min("Age").alias("min_age"),
            max("Age").alias("max_age")
        ).collect()
        
        stats_list = [
            {
                "Department": row["Department"],
                "count": row["count"],
                "avg_salary": float(row["avg_salary"]),
                "total_salary": float(row["total_salary"]),
                "min_salary": float(row["min_salary"]),
                "max_salary": float(row["max_salary"]),
                "min_age": int(row["min_age"]),
                "max_age": int(row["max_age"])
            }
            for row in stats
        ]
        return jsonify(stats_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/filter_employees')
def filter_employees():
    try:
        department = request.args.get('department')
        min_age = request.args.get('min_age', type=int)
        max_age = request.args.get('max_age', type=int)
        min_salary = request.args.get('min_salary', type=int)
        max_salary = request.args.get('max_salary', type=int)
        
        df = load_employee_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
            
        if department:
            df = df.filter(col("Department") == department)
        if min_age:
            df = df.filter(col("Age") >= min_age)
        if max_age:
            df = df.filter(col("Age") <= max_age)
        if min_salary:
            df = df.filter(col("Salary") >= min_salary)
        if max_salary:
            df = df.filter(col("Salary") <= max_salary)
        
        filtered_employees = df.collect()
        employee_list = [
            {
                "Emp_No": row["Emp_No"],
                "Emp_Name": row["Emp_Name"],
                "Salary": row["Salary"],
                "Age": row["Age"],
                "Department": row["Department"]
            }
            for row in filtered_employees
        ]
        return jsonify(employee_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/export_employees')
def export_employees():
    try:
        department = request.args.get('department')
        min_age = request.args.get('min_age', type=int)
        max_age = request.args.get('max_age', type=int)
        min_salary = request.args.get('min_salary', type=int)
        max_salary = request.args.get('max_salary', type=int)
        
        df = load_employee_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
            
        if department:
            df = df.filter(col("Department") == department)
        if min_age:
            df = df.filter(col("Age") >= min_age)
        if max_age:
            df = df.filter(col("Age") <= max_age)
        if min_salary:
            df = df.filter(col("Salary") >= min_salary)
        if max_salary:
            df = df.filter(col("Salary") <= max_salary)
        
        filtered_df = df.toPandas()
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype='text/csv',
            as_attachment=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
