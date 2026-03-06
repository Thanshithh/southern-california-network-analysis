import pandas as pd
import pyodbc
import os
import sys

# Path to your file
# The file is located at data/.ipynb_checkpoints/XXH2023_YRBS_Data.mdb
# We need the absolute path for the driver
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up from results/visualizations to root
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
db_file = os.path.join(project_root, "data", ".ipynb_checkpoints", "XXH2023_YRBS_Data.mdb")

print(f"Looking for database at: {db_file}")

if not os.path.exists(db_file):
    print("Error: File not found!")
    sys.exit(1)

# Connect
# NOTE: This driver string typically only works on Windows. 
# On macOS, you usually need 'mdbtools' or a specifically configured ODBC driver.
driver_str = 'Microsoft Access Driver (*.mdb, *.accdb)'
conn_str = f'DRIVER={{{driver_str}}};DBQ={db_file};'

try:
    conn = pyodbc.connect(conn_str)
    print("Successfully connected to the database.")
    
    # List tables
    cursor = conn.cursor()
    tables = cursor.tables(tableType='TABLE')

    print("Tables found:")
    for t in tables:
        print(t.table_name)
        
except pyodbc.Error as e:
    print("\n❌ RETURNED ERROR:", e)
    print("\n⚠️  EXPLANATION:")
    print("   The 'Microsoft Access Driver' is built into Windows but NOT macOS.")
    print("   To read MDB files on Mac, you typically need to install 'mdbtools'.")
    print("   Since 'brew' is missing, installing mdbtools is difficult.")
except Exception as e:
    print(f"An error occurred: {e}")
