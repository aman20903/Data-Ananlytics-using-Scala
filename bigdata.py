import findspark
import warnings

def warn(*args, **kwargs):
    pass

# Suppress generated warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Data Analysis using Spark") \
    .getOrCreate()

# Read data from the "emp" CSV file and import it into a DataFrame variable named "employees_df"
employees_df = spark.read.csv("employees.csv", header=True, inferSchema=True)

# Display the dataframe content
employees_df.show()

# Define a Schema for the input data and read the file using the user-defined Schema
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

schema = StructType([
    StructField("Emp_No", IntegerType(), False),
    StructField("Emp_Name", StringType(), False),
    StructField("Salary", IntegerType(), False),
    StructField("Age", IntegerType(), False),
    StructField("Department", StringType(), False),
])

# Create a dataframe on top of a CSV file
employees_df = (spark.read
    .format("csv")
    .schema(schema)
    .option("header", "true")
    .load("employees.csv")
)

# Display the dataframe content
employees_df.show()

# Display all columns of the DataFrame, along with their respective data types
employees_df.printSchema()

# Create a temporary view named "employees" for the DataFrame
employees_df.createOrReplaceTempView("employees")

# SQL query to fetch solely the records from the View where the age exceeds 30
spark.sql("SELECT * FROM employees WHERE Age > 30").show()

# SQL query to calculate the average salary of employees grouped by department
spark.sql(
    "SELECT Department, AVG(Salary) AS Avg_Salary "
    "FROM employees "
    "GROUP BY Department"
).show()

# Apply a filter to select records where the department is 'IT'
employees_df.filter(employees_df["Department"] == "IT").show()

from pyspark.sql.functions import col

# Add a new column "SalaryAfterBonus" with 10% bonus added to the original salary
employees_df.withColumn("SalaryAfterBonus", col("Salary") * 1.1).show()

from pyspark.sql.functions import max

# Group data by age and calculate the maximum salary for each age group
employees_df.groupby(["Age"]) \
    .agg(max("Salary").alias("Max_Salary")) \
    .sort("Age") \
    .show()

# Join the DataFrame with itself based on the "Emp_No" column
employees_df.join(employees_df, "Emp_No", "inner").show()

# Calculate the average age of employees
from pyspark.sql.functions import avg

employees_df.agg(avg("Age").alias("Avg_Age")).show()

# Calculate the total salary for each department.
# Hint - Use GroupBy and Aggregate functions
from pyspark.sql.functions import sum

employees_df.groupBy("Department") \
    .agg(sum("Salary") \
    .alias("Total_Salary")) \
    .show()

# Sort the DataFrame by age in ascending order and then by salary
# in descending order
employees_df.sort(["Age", "Salary"], ascending=[True, False]).show()

from pyspark.sql.functions import count

# Calculate the number of employees in each department
employees_df.groupBy("Department") \
    .agg(count("*").alias("Emp_Count")) \
    .show()

# Apply a filter to select records where the employee's name
# contains the letter 'o'
employees_df.filter(col("Emp_Name").contains("o")).show()

spark.stop()